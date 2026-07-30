"""Central store: modules, cameras, events.

Plain stdlib ``sqlite3`` on purpose. Central's whole value is being light enough
to deploy anywhere without a GPU, and an ORM plus an async driver would be two
dependencies bought for three tables. Swap in Postgres when a single box stops
being enough to hold the fleet's events — the queries here are ordinary SQL.

All access goes through :class:`Store`, which serialises writes behind a lock and
is safe to call from FastAPI's threadpool. Reads use ``row_factory`` so callers
get dicts, not tuples.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,          -- how central and browsers reach it
    gpu           TEXT,
    max_feeds     INTEGER NOT NULL DEFAULT 0,
    fps_budget    REAL    NOT NULL DEFAULT 0,   -- measured aggregate detect fps
    version       TEXT,
    registered_at REAL NOT NULL,
    last_seen     REAL NOT NULL,
    active_feeds  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    module_id   TEXT,                     -- NULL = not placed yet
    feed_id     TEXT,                     -- the module's own id for this feed
    status      TEXT NOT NULL DEFAULT 'pending',
    geometry    TEXT,                     -- JSON: zone_polygon/line_start/line_end
    source_fps  REAL NOT NULL DEFAULT 30,
    pinned_module TEXT,                   -- user pinned this camera to one module
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    module_id   TEXT,
    kind        TEXT NOT NULL,            -- phone_usage | zone_intrusion | ...
    person_id   INTEGER,
    frame_number INTEGER,
    timestamp   TEXT,                     -- stream-relative HH:MM:SS.mmm
    confidence  REAL,
    received_at REAL NOT NULL,
    payload     TEXT                      -- full original event JSON
);

CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_events_recent ON events(id DESC);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")     # concurrent reads
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._migrate()
            self._db.commit()

    def _migrate(self) -> None:
        """Add columns missing from an older database.

        `CREATE TABLE IF NOT EXISTS` silently does nothing when the table already
        exists, so new columns never appear on an upgrade. Until there's a real
        migration tool, additive ALTERs guarded by the existing column list keep an
        already-deployed DB working instead of failing on first query.
        """
        cols = {r["name"] for r in
                self._db.execute("PRAGMA table_info(cameras)").fetchall()}
        for name, decl in (("pinned_module", "TEXT"),):
            if name not in cols:
                self._db.execute(f"ALTER TABLE cameras ADD COLUMN {name} {decl}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------- modules
    def upsert_module(self, mod: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                """INSERT INTO modules (id, url, gpu, max_feeds, fps_budget, version,
                                        registered_at, last_seen, active_feeds)
                   VALUES (?,?,?,?,?,?,?,?,0)
                   ON CONFLICT(id) DO UPDATE SET
                     url=excluded.url, gpu=excluded.gpu,
                     max_feeds=excluded.max_feeds, fps_budget=excluded.fps_budget,
                     version=excluded.version, last_seen=excluded.last_seen""",
                (mod["id"], mod["url"], mod.get("gpu"), int(mod.get("max_feeds", 0)),
                 float(mod.get("fps_budget", 0)), mod.get("version"), now, now),
            )
            self._db.commit()

    def touch_module(self, module_id: str, active_feeds: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE modules SET last_seen=?, active_feeds=? WHERE id=?",
                (time.time(), int(active_feeds), module_id),
            )
            self._db.commit()

    def modules(self, stale_after: float = 30.0) -> List[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            rows = self._db.execute("SELECT * FROM modules ORDER BY id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["online"] = (now - d["last_seen"]) <= stale_after
            d["seconds_since_seen"] = round(now - d["last_seen"], 1)
            out.append(d)
        return out

    def module(self, module_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._db.execute("SELECT * FROM modules WHERE id=?",
                                 (module_id,)).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------- cameras
    def add_camera(self, name: str, url: str, geometry: Optional[dict] = None,
                   pinned_module: Optional[str] = None) -> str:
        """Register a camera. `pinned_module` forces it onto one module."""
        cam_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._db.execute(
                """INSERT INTO cameras (id, name, url, geometry, pinned_module,
                                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (cam_id, name, url,
                 json.dumps(geometry) if geometry else None, pinned_module, now, now),
            )
            self._db.commit()
        return cam_id

    def cameras(self, module_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM cameras"
        args: tuple = ()
        if module_id:
            sql += " WHERE module_id=?"
            args = (module_id,)
        sql += " ORDER BY created_at"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._camera_row(r) for r in rows]

    def camera(self, cam_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._db.execute("SELECT * FROM cameras WHERE id=?",
                                 (cam_id,)).fetchone()
        return self._camera_row(r) if r else None

    @staticmethod
    def _camera_row(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d["geometry"] = json.loads(d["geometry"]) if d.get("geometry") else None
        return d

    def assign_camera(self, cam_id: str, module_id: Optional[str],
                      feed_id: Optional[str], status: str) -> None:
        with self._lock:
            self._db.execute(
                """UPDATE cameras SET module_id=?, feed_id=?, status=?, updated_at=?
                   WHERE id=?""",
                (module_id, feed_id, status, time.time(), cam_id),
            )
            self._db.commit()

    def set_camera_status(self, cam_id: str, status: str) -> None:
        with self._lock:
            self._db.execute("UPDATE cameras SET status=?, updated_at=? WHERE id=?",
                             (status, time.time(), cam_id))
            self._db.commit()

    def set_camera_geometry(self, cam_id: str, geometry: Optional[dict]) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE cameras SET geometry=?, updated_at=? WHERE id=?",
                (json.dumps(geometry) if geometry else None, time.time(), cam_id),
            )
            self._db.commit()

    def delete_camera(self, cam_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
            self._db.commit()

    def unplace_module_cameras(self, module_id: str) -> List[str]:
        """Detach every camera from a module (it died). Returns their ids."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM cameras WHERE module_id=?",
                                    (module_id,)).fetchall()
            ids = [r["id"] for r in rows]
            self._db.execute(
                """UPDATE cameras SET module_id=NULL, feed_id=NULL,
                   status='pending', updated_at=? WHERE module_id=?""",
                (time.time(), module_id),
            )
            self._db.commit()
        return ids

    # -------------------------------------------------------------- events
    def add_events(self, camera_id: str, module_id: Optional[str],
                   events: List[dict]) -> int:
        """Insert a batch. Returns how many rows landed."""
        if not events:
            return 0
        now = time.time()
        rows = [
            (camera_id, module_id,
             e.get("event") or e.get("kind") or "unknown",
             e.get("person_id"), e.get("frame_number"), e.get("timestamp"),
             e.get("confidence"), now, json.dumps(e))
            for e in events
        ]
        with self._lock:
            self._db.executemany(
                """INSERT INTO events (camera_id, module_id, kind, person_id,
                                       frame_number, timestamp, confidence,
                                       received_at, payload)
                   VALUES (?,?,?,?,?,?,?,?,?)""", rows,
            )
            self._db.commit()
        return len(rows)

    def recent_events(self, limit: int = 100,
                      camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = ("SELECT e.*, c.name AS camera_name FROM events e "
               "LEFT JOIN cameras c ON c.id = e.camera_id")
        args: list = []
        if camera_id:
            sql += " WHERE e.camera_id=?"
            args.append(camera_id)
        sql += " ORDER BY e.id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, tuple(args)).fetchall()
        return [dict(r) for r in rows]

    def event_counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, COUNT(*) AS n FROM events GROUP BY kind").fetchall()
        return {r["kind"]: r["n"] for r in rows}
