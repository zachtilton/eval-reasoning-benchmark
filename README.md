# Evaluative Reasoning LLM Benchmark

Dissertation research codebase: testing whether LLMs can replicate expert evaluative reasoning on UN evaluation report conclusions.

## Directory Structure
code README.md
# Evaluative Reasoning LLM Benchmark

Dissertation research codebase: testing whether LLMs can replicate expert evaluative reasoning on UN evaluation report conclusions.

## Directory Structure

    eval-reasoning-benchmark/
    ├── config/              # API configuration, model parameters
    │   └── prompts/         # Zero-shot and few-shot prompt templates
    ├── src/
    │   ├── extraction/      # Phase 1: fragment extraction support
    │   ├── api_client/      # G.1-G.3: API calls, retry, parsing
    │   ├── scoring/         # G.4-G.6: coherence, accuracy, adjudication
    │   └── analysis/        # H-series: stats, visualization
    ├── data/
    │   ├── raw/             # Fragment corpus
    │   ├── gold_standard/   # Locked expert judgments
    │   └── responses/       # Model outputs
    ├── logs/                # Execution and session logs
    ├── appendices/          # Templates from E.1-E.4, D.1-D.3
    └── tests/               # Unit tests for scoring logic

## Author

Zach Tilton — Interdisciplinary Ph.D. in Evaluation, Western Michigan University
```


