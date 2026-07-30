"""Module -> central reporting: register, heartbeat, and deliver events.

Deploying to another GPU box is entirely a matter of three env vars; nothing on
central is edited, because the module is what makes contact:

    CENTRAL_URL=https://central.yourorg.internal
    MODULE_TOKEN=<shared secret>
    MODULE_PUBLIC_URL=http://10.0.1.22:8001   # how central/browsers reach THIS box

The module always initiates. That is not incidental — outbound HTTPS works from
behind NAT and a corporate firewall, whereas central dialling in usually does not.
Today central does call back to the module's ``/feeds/*`` API (every host is
publicly reachable), so the two directions coexist; if a module later lands behind
NAT, the fix is to carry commands down a socket opened from here rather than
redesigning the contract.

**Event delivery is durable.** Violations are the product, so events are spooled
to disk first and only removed once central has acknowledged them. A network blip
delays alerts; it must never lose them. Duplicates are possible after a retry —
that is the deliberate trade (at-least-once, not at-most-once).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("operator_monitor")

HEARTBEAT_SECONDS = 10.0
SPOOL_FLUSH_SECONDS = 2.0
MAX_BATCH = 200


def _post(url: str, body: dict, token: str, timeout: float = 10.0) -> Any:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Module-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


class EventSpool:
    """Append-only disk queue for events awaiting central.

    One JSON object per line. Nothing is deleted until central has confirmed it,
    so a crash or an unreachable central costs latency, not data.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_batch(self, limit: int = MAX_BATCH) -> List[dict]:
        if not self.path.exists():
            return []
        out: List[dict] = []
        with self._lock:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue          # skip a torn line, keep the rest
                    if len(out) >= limit:
                        break
        return out

    def drop(self, count: int) -> None:
        """Remove the first `count` records — only after central acknowledged."""
        if count <= 0 or not self.path.exists():
            return
        with self._lock:
            with self.path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
            rest = lines[count:]
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                fh.writelines(rest)
            tmp.replace(self.path)

    def pending(self) -> int:
        if not self.path.exists():
            return 0
        with self._lock:
            with self.path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())


class CentralReporter:
    """Registers this module with central, heartbeats, and ships events.

    Runs on its own daemon thread so nothing here can block the event loop that is
    serving feeds. If central is absent (``CENTRAL_URL`` unset) every method is a
    no-op, so the module still runs perfectly well standalone.
    """

    def __init__(
        self,
        cfg,
        *,
        central_url: Optional[str] = None,
        module_id: Optional[str] = None,
        public_url: Optional[str] = None,
        token: Optional[str] = None,
        spool_path: Optional[Path] = None,
        status_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._cfg = cfg
        self.central = (central_url or os.environ.get("CENTRAL_URL") or "").rstrip("/")
        self.module_id = (
            module_id or os.environ.get("MODULE_ID") or f"module-{uuid.uuid4().hex[:8]}"
        )
        self.public_url = (
            public_url or os.environ.get("MODULE_PUBLIC_URL") or ""
        ).rstrip("/")
        # Where the SEPARATE raw video service is reachable. Advertised so central
        # can hand browsers a raw-playback URL; blank means this host offers
        # analysed video only and the wall falls back to the detection stream.
        self.raw_url = (os.environ.get("RAW_PUBLIC_URL") or "").rstrip("/")
        self.token = token or os.environ.get("MODULE_TOKEN") or ""
        self._status = status_provider or (lambda: {"active_feeds": 0, "feeds": []})
        # Spool file is namespaced by module id. Two module instances started from
        # the same checkout (e.g. one per GPU on one box) would otherwise share one
        # spool — and `drop()` rewrites the whole file, so one instance deleting its
        # delivered batch would silently destroy the other's undelivered events.
        base = Path(getattr(cfg, "events_csv", Path("output/events.csv"))).parent
        self.spool = EventSpool(
            spool_path or (base / f"event-spool-{self.module_id}.jsonl")
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._registered = False

    @property
    def enabled(self) -> bool:
        return bool(self.central)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "CentralReporter":
        if not self.enabled:
            logger.info("CENTRAL_URL not set — running standalone, no reporting.")
            return self
        if not self.public_url:
            logger.warning(
                "MODULE_PUBLIC_URL not set; central cannot reach this module to "
                "assign cameras or hand out video URLs. Set it to this host's "
                "reachable address."
            )
        self._thread = threading.Thread(target=self._loop, name="central-reporter",
                                        daemon=True)
        self._thread.start()
        logger.info("Reporting to central %s as module '%s' (public=%s).",
                    self.central, self.module_id, self.public_url or "unset")
        return self

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------------- events
    def report_events(self, camera_id: str, events: List[dict]) -> None:
        """Queue violations for delivery. Returns immediately (disk append only)."""
        if not self.enabled or not events:
            return
        self.spool.append({
            "camera_id": camera_id,
            "module_id": self.module_id,
            "events": events,
            "queued_at": time.time(),
        })

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        last_beat = 0.0
        while not self._stop.is_set():
            try:
                if not self._registered:
                    self._register()
                if self._registered:
                    now = time.time()
                    if now - last_beat >= HEARTBEAT_SECONDS:
                        self._heartbeat()
                        last_beat = now
                    self._flush_spool()
            except Exception:  # noqa: BLE001 - reporting must never kill the module
                logger.debug("reporter iteration failed", exc_info=True)
            self._stop.wait(SPOOL_FLUSH_SECONDS)

    def _register(self) -> None:
        body = {
            "id": self.module_id,
            "url": self.public_url,
            "gpu": self._gpu_name(),
            "raw_url": self.raw_url,
            "max_feeds": int(getattr(self._cfg, "max_feeds", 0)),
            "fps_budget": float(getattr(self._cfg, "fps_budget", 0) or 0),
            "version": "module/1",
        }
        try:
            _post(f"{self.central}/api/modules/register", body, self.token)
            self._registered = True
            logger.info("Registered with central as '%s'.", self.module_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Register failed (%s); retrying.", exc)

    def _heartbeat(self) -> None:
        try:
            res = _post(f"{self.central}/api/modules/{self.module_id}/heartbeat",
                        self._status(), self.token)
            if isinstance(res, dict) and res.get("reregister"):
                self._registered = False
        except urllib.error.HTTPError as exc:
            if exc.code == 409:                 # central forgot us
                self._registered = False
        except Exception:  # noqa: BLE001
            logger.debug("heartbeat failed", exc_info=True)

    def _flush_spool(self) -> None:
        """Ship spooled events; only drop what central confirms."""
        batch = self.spool.read_batch()
        if not batch:
            return
        sent = 0
        for record in batch:
            try:
                _post(f"{self.central}/api/events", record, self.token)
                sent += 1
            except Exception:  # noqa: BLE001 - stop at the first failure; retry later
                break
        if sent:
            self.spool.drop(sent)
            remaining = self.spool.pending()
            if remaining:
                logger.info("Delivered %d event batch(es); %d still spooled.",
                            sent, remaining)

    @staticmethod
    def _gpu_name() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
