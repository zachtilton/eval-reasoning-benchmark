"""
src/pilot — pre-run cost/token audit for the evaluative reasoning benchmark.

Module layout
-------------
fragments.py             — fragments.jsonl loading, id/text adaptation
calibration.py            — calibration-example token measurement
task1_corpus_audit.py     — Task 1: corpus + scaffold token audit
task2_pilot_batch.py      — Task 2: pinned-fragment pilot API batch
task3_cost_projection.py  — Task 3: updated 5,400-call cost projection

Never mixed into the real 5,400-call dataset — see each task module's
docstring for its isolated output paths. Pilot responses are for
cost/token estimation only: never scored, never compared to gold-standard
judgments, never included in reliability checks (RC1-RC3).

Public interface
----------------
run_task1(...)  — corpus/scaffold token audit
run_task2(...)  — pinned-fragment pilot API batch
run_task3(...)  — updated cost projection (optionally chains Task 1 -> 2 -> 3)
"""

from .task1_corpus_audit import run_task1
from .task2_pilot_batch import run_task2
from .task3_cost_projection import run_task3

__all__ = [
    "run_task1",
    "run_task2",
    "run_task3",
]
