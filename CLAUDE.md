# Project Context: Evaluative Reasoning LLM Benchmark

## What This Is
Dissertation research codebase for testing LLM evaluative reasoning capacity.
150 UN evaluation fragments assessed by human expert and 6 LLM models under
zero-shot and few-shot conditions. 5,400 total API calls.

## Key Design Decisions
- GPT 5.2 is the primary diagnostic model; 5 others are comparison only
- Coherence validation uses tiered approach: rule-based keywords (Tier 1),
  then Claude Haiku at temp 0 (Tier 2) for ambiguous cases
- Exactly 4 calibration examples (2 sound, 2 not sound)
- Only Type 1 failures (coherent but wrong) receive failure mode coding
- RC2 validates automated coherence pipeline against human judgment
- Framework/template analysis, not grounded theory
- 22-checkpoint Integrated Evaluative Reasoning Checklist across 6 domains

## Repo Structure
See README.md for directory layout.

## Code Standards
- Python 3.10+
- All functions need docstrings
- Config separated from logic
- Reproducibility is paramount: random seeds, parameter logging, hash verification
- Budget constraint: ~$500 for ~5,400 API calls
