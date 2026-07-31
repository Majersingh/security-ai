"""CrowdGatheringRule: grouping, the allowance knob, and the hold debounce.

No GPU and no model — the rule is pure geometry over person boxes, so we feed it
synthetic detections directly.

    python tests/test_crowd.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "module" / "core"))

import numpy as np
import supervision as sv

from behavior import BehaviorEngine, CrowdGatheringRule
from config import Config
from events import EventLog, SnapshotManager


def persons(*boxes) -> sv.Detections:
    """Tracked person detections from (x1, y1, x2, y2) tuples."""
    xyxy = np.array(boxes, dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(len(boxes), 0.9, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.arange(1, len(boxes) + 1),
    )


def standing_at(*xs, y=100, w=60, h=200):
    """People of equal height standing at the given x positions."""
    return persons(*[(x, y, x + w, y + h) for x in xs])


EMPTY = sv.Detections.empty()
checks = []


def check(name: str, got, want) -> None:
    ok = got == want
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")


# --------------------------------------------------------------- grouping

cfg = Config()
cfg.crowd_max_persons = 3          # up to 3 together is fine
cfg.crowd_proximity_factor = 0.6   # gap < 0.6 * height (=120 px) groups them
rule = CrowdGatheringRule(cfg)

# 3 people shoulder to shoulder = the allowance exactly, no event.
check("3 together (allowed=3)", len(rule.evaluate(standing_at(0, 70, 140), EMPTY)), 0)

# A 4th joins the huddle -> all four are flagged.
check("4 together", len(rule.evaluate(standing_at(0, 70, 140, 210), EMPTY)), 4)

# Same 4 people, but spread far apart (600 px gaps >> 120) -> no group at all.
check("4 spread out", len(rule.evaluate(standing_at(0, 700, 1400, 2100), EMPTY)), 0)

# Two separate huddles of 2 each: neither exceeds the allowance.
check("2 + 2 apart", len(rule.evaluate(standing_at(0, 70, 2000, 2070), EMPTY)), 0)

# Transitive closure: a chain of 5, each 100 px from the next (under 120), is ONE
# group of 5 even though the ends are 640 px apart.
check("chain of 5", len(rule.evaluate(standing_at(0, 160, 320, 480, 640), EMPTY)), 5)

# The allowance is the knob: same 4 people, allowance raised to 4 -> silent.
cfg4 = Config()
cfg4.crowd_max_persons = 4
check("4 together (allowed=4)",
      len(CrowdGatheringRule(cfg4).evaluate(standing_at(0, 70, 140, 210), EMPTY)), 0)

# Distance scales with body height, not pixels: two people 150 px apart are a group
# when they are tall (near camera, h=400 -> threshold 240) but not when they are
# small (far away, h=100 -> threshold 60).
cfg2 = Config()
cfg2.crowd_max_persons = 1
near = CrowdGatheringRule(cfg2).evaluate(
    persons((0, 0, 60, 400), (210, 0, 270, 400)), EMPTY)          # gap 150, h 400
far = CrowdGatheringRule(cfg2).evaluate(
    persons((0, 0, 60, 100), (210, 0, 270, 100)), EMPTY)          # gap 150, h 100
check("near camera, 150px gap", len(near), 2)
check("far away, 150px gap", len(far), 0)

# The group box spans the whole gathering (that is what gets snapshotted).
obs = CrowdGatheringRule(cfg).evaluate(standing_at(0, 70, 140, 210), EMPTY)
check("focus box spans group", tuple(obs[0].focus_box), (0.0, 100.0, 270.0, 300.0))


# ---------------------------------------------------------------- debounce

# crowd_hold_seconds must gate the event, NOT violation_start_seconds.
cfg_t = Config()
cfg_t.crowd_max_persons = 3
cfg_t.violation_start_seconds = 1.0     # phone timing: 10 frames at 10 fps
cfg_t.crowd_hold_seconds = 3.0          # crowd timing: 30 frames at 10 fps
cfg_t.crowd_clear_seconds = 1.0

log = EventLog(cfg_t)
engine = BehaviorEngine(
    cfg_t, [CrowdGatheringRule(cfg_t)], log, SnapshotManager(cfg_t), fps=10.0
)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
crowd = standing_at(0, 70, 140, 210)

for i in range(29):
    engine.process(crowd, EMPTY, i, frame)
check("no event before hold elapses", len(log._events), 0)

engine.process(crowd, EMPTY, 29, frame)
check("event once hold elapses", len(log._events), 4)   # one per person
check("event label", log._events[0].event, "Crowd Gathering")

# Dispersing for crowd_clear_seconds closes the episode; regrouping re-fires.
for i in range(30, 45):
    engine.process(standing_at(0, 700, 1400, 2100), EMPTY, i, frame)
for i in range(45, 75):
    engine.process(crowd, EMPTY, i, frame)
check("re-fires after dispersing", len(log._events), 8)

print(f"\n{sum(checks)}/{len(checks)} passed")
sys.exit(0 if all(checks) else 1)
