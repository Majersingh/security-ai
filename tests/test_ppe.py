"""HelmetComplianceRule: association, the suppressors, and the hold debounce.

No GPU and no PPE model — the rule is geometry over boxes, so we feed it
synthetic detections with a helmet class id.

The claim is always an absence — "no helmet on this head" — so much of what is
asserted here is that the rule STAYS SILENT when the evidence is too weak.

    python tests/test_ppe.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "module" / "core"))

import numpy as np
import supervision as sv

from behavior import (
    BehaviorEngine, HelmetComplianceRule, PhoneUsageRule, build_rules,
)
from config import Config
from events import EventLog, SnapshotManager
from tracker import Tracker

HELMET = 1
PHONE = 67
HEAD = 3


def cfg_ppe(**over) -> Config:
    """No head class: the coarse fallback (helmet somewhere in the top third)."""
    c = Config()
    c.ppe_enabled = True
    c.helmet_class_id = HELMET
    for k, v in over.items():
        setattr(c, k, v)
    return c


def cfg_positive(**over) -> Config:
    """With a head class: helmet/head position matching."""
    return cfg_ppe(head_class_id=HEAD, **over)


def person(box, conf=0.9, tid=1) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([box], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
        class_id=np.zeros(1, dtype=int),
        tracker_id=np.array([tid]),
    )


def dets(class_id, boxes, conf=0.8):
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.full(len(boxes), conf, dtype=np.float32),
        class_id=np.full(len(boxes), class_id),
    )


def objs(class_id=HELMET, *boxes, extra=None):
    """An objects dict with the given boxes under `class_id` (empty otherwise).

    ``extra`` is ``{class_id: [boxes]}`` for additional classes in the same frame.
    """
    empty = sv.Detections.empty()
    table = {HELMET: empty, PHONE: empty, HEAD: empty}
    if boxes:
        table[class_id] = dets(class_id, boxes)
    for cid, more in (extra or {}).items():
        table[int(cid)] = dets(int(cid), more)
    return table


checks = []


def check(name, got, want) -> None:
    ok = got == want
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")


# A person 300 px tall, well inside the frame: judgeable by every suppressor.
# head region = top 35% + 10% margin above -> y 70..205
WORKER = (100.0, 100.0, 160.0, 400.0)

rule = HelmetComplianceRule(cfg_ppe())

# ------------------------------------------------------------- association

check("no helmet anywhere -> violation",
      len(rule.evaluate(person(WORKER), objs())), 1)

check("helmet on head -> compliant",
      len(rule.evaluate(person(WORKER), objs(HELMET, (110, 80, 150, 110)))), 0)

# The margin matters: a correctly-worn helmet often sits ENTIRELY above the
# person box, which the detector clips at the scalp. Without ppe_head_margin
# this helmet would fall outside the search region and the worker be accused.
check("helmet above box top -> compliant",
      len(rule.evaluate(person(WORKER), objs(HELMET, (110, 75, 150, 99)))), 0)

# A helmet in frame but not on this person's head is not compliance.
check("helmet at feet -> violation",
      len(rule.evaluate(person(WORKER), objs(HELMET, (110, 350, 150, 380)))), 1)

# A helmet belonging to someone standing beside them must not count either.
check("helmet beside person -> violation",
      len(rule.evaluate(person(WORKER), objs(HELMET, (400, 80, 440, 110)))), 1)

# A phone on the frame is not a helmet.
check("phone is not a helmet",
      len(rule.evaluate(person(WORKER), objs(PHONE, (110, 80, 150, 110)))), 1)


# -------------------------------------------------------------- suppressors
# Each of these has NO helmet, so the violation path would fire; they must be
# skipped as "cannot judge" instead.

check("too far away (90px tall) -> skipped",
      len(rule.evaluate(person((100, 100, 130, 190)), objs())), 0)

check("weak person detection -> skipped",
      len(rule.evaluate(person(WORKER, conf=0.3), objs())), 0)

check("head cropped by top edge -> skipped",
      len(rule.evaluate(person((100, 4, 160, 304)), objs())), 0)

# ...and the suppressors are knobs, not hard-coded: lower the height floor and
# the distant worker becomes judgeable again.
check("height floor lowered -> judged",
      len(HelmetComplianceRule(cfg_ppe(ppe_min_person_height=50))
          .evaluate(person((100, 100, 130, 190)), objs())), 1)

# Confidence reported is the PERSON's score, not a fabricated 1.0 — there is no
# helmet detection to score.
obs = rule.evaluate(person(WORKER, conf=0.77), objs())
check("confidence is the person score", round(obs[0].confidence, 2), 0.77)


# ------------------------------------------- head class: helmet/head POSITION match
# Heads are labelled on EVERY worker, covered or not; helmets wherever they are.
# The rule compares positions — a helmet ON the head is compliance.

pos = HelmetComplianceRule(cfg_positive())
HEAD_BOX = (110.0, 80.0, 150.0, 115.0)          # a head inside the head region
WORN = (110.0, 78.0, 150.0, 108.0)              # helmet overlapping that head

check("head with no helmet -> violation",
      len(pos.evaluate(person(WORKER), objs(HEAD, HEAD_BOX))), 1)

check("head + helmet on it -> compliant",
      len(pos.evaluate(person(WORKER), objs(HELMET, WORN,
                                            extra={HEAD: [HEAD_BOX]}))), 0)

# CARRYING is not wearing. The estimated region spans the top third of the person,
# which on a standing worker reaches the chest, so a held helmet lands inside it —
# but never on the head box. Matching against the head box is what catches this.
CHEST_HELMET = (105.0, 175.0, 145.0, 205.0)     # inside head region (70..205)
check("helmet carried at chest -> violation",
      len(pos.evaluate(person(WORKER), objs(HELMET, CHEST_HELMET,
                                            extra={HEAD: [HEAD_BOX]}))), 1)

# The same input WITHOUT a head class is read as compliant: nothing tight to aim
# at, so the rule can only ask "is a helmet somewhere up there?". Asserted so the
# limitation stays visible — it is the clearest reason to train the head class.
check("no head class is fooled by a carried helmet",
      len(HelmetComplianceRule(cfg_ppe()).evaluate(
          person(WORKER), objs(HELMET, CHEST_HELMET))), 0)

# Head not visible (facing into machinery, occluded) -> nothing to judge.
check("no head detected -> silent",
      len(pos.evaluate(person(WORKER), objs())), 0)

# Helmet detected but the head under it missed. Silent, which is the right OUTPUT
# for a compliant worker — but it is silence from "cannot judge", not from
# verified compliance. The same silence hides a real violation when a BARE head
# goes undetected, so this mode's failure direction is missed violations, never
# false accusations. Systematic misses here mean the dataset lacks head boxes on
# helmeted workers (see the labelling rule in training/ppe.yaml).
check("helmet detected, head missed -> silent",
      len(pos.evaluate(person(WORKER), objs(HELMET, WORN))), 0)

# Someone else's head, across the frame, is not this person's violation.
check("head elsewhere -> not this person",
      len(pos.evaluate(person(WORKER), objs(HEAD, (400, 80, 440, 115)))), 0)

# The suppressors still apply: the claim rests on a MISSING helmet, so a worker
# too far away for the model to resolve one must not be accused.
check("distant worker suppressed",
      len(pos.evaluate(person((100, 100, 130, 190)),
                       objs(HEAD, (105, 85, 125, 105)))), 0)
check("weak person detection suppressed",
      len(pos.evaluate(person(WORKER, conf=0.3), objs(HEAD, HEAD_BOX))), 0)

# Confidence comes from the head detection, not the person.
obs_pos = pos.evaluate(person(WORKER, conf=0.9),
                       {HELMET: sv.Detections.empty(), PHONE: sv.Detections.empty(),
                        HEAD: dets(HEAD, [HEAD_BOX], conf=0.66)})
check("confidence is the head score", round(obs_pos[0].confidence, 2), 0.66)

# Wiring: the head class must reach the detector and the router.
check("head class kept by detector", cfg_positive().detect_class_ids(), [0, 1, 3, 67])
check("head class named on the wire",
      cfg_positive().object_class_names()[HEAD], "head")


# ----------------------------------------------------------------- wiring

check("rule off without a PPE model",
      [r.name for r in build_rules(Config())].count("no_helmet"), 0)
check("rule on when ppe_enabled",
      [r.name for r in build_rules(cfg_ppe())].count("no_helmet"), 1)

# The detector must be told to return helmets, or the rule sees nothing at all.
check("helmet class kept by detector", cfg_ppe().detect_class_ids(), [0, 1, 67])
check("helmet class dropped when off", Config().detect_class_ids(), [0, 67])

# Routing keys objects by class id, with an entry per configured class even when
# nothing was detected, so rules can index without guarding.
tracked = sv.Detections(
    xyxy=np.array([[100, 100, 160, 400], [110, 80, 150, 110]], dtype=np.float32),
    confidence=np.array([0.9, 0.8], dtype=np.float32),
    class_id=np.array([0, HELMET]),
    tracker_id=np.array([1, 2]),
)
persons, table = Tracker(cfg_ppe()).route(tracked)
check("routes persons", len(persons), 1)
check("routes helmets by class id", len(table[HELMET]), 1)
check("empty entry for undetected class", len(table[PHONE]), 0)
check("object identities stripped", table[HELMET].tracker_id, None)

# Regression: phone usage still works now that it reads from the dict.
phone_obs = PhoneUsageRule(Config()).evaluate(
    person(WORKER), objs(PHONE, (120, 200, 140, 230)))
check("phone rule via objects dict", len(phone_obs), 1)


# ------------------------------------------------------------------ debounce

# The strongest suppressor: a helmet must be absent CONTINUOUSLY. One dropped
# detection in a 5 s window must not produce an event.
c = cfg_ppe(ppe_hold_seconds=5.0, ppe_clear_seconds=2.0)
log = EventLog(c)
engine = BehaviorEngine(
    c, [HelmetComplianceRule(c)], log, SnapshotManager(c), fps=10.0
)   # 5.0 s at 10 fps = 50 frames
frame = np.zeros((480, 640, 3), dtype=np.uint8)
bare, helmeted = objs(), objs(HELMET, (110, 80, 150, 110))

for i in range(49):
    engine.process(person(WORKER), bare, i, frame)
check("no event before hold elapses", len(log._events), 0)

engine.process(person(WORKER), bare, 49, frame)
check("event once hold elapses", len(log._events), 1)
check("event label", log._events[0].event, "No Helmet (PPE)")

# A single frame where the helmet IS seen resets nothing mid-episode, but a
# sustained clear (2 s = 20 frames) closes it and a fresh violation re-fires.
for i in range(50, 75):
    engine.process(person(WORKER), helmeted, i, frame)
for i in range(75, 125):
    engine.process(person(WORKER), bare, i, frame)
check("re-fires after wearing then removing", len(log._events), 2)

print(f"\n{sum(checks)}/{len(checks)} passed")
sys.exit(0 if all(checks) else 1)
