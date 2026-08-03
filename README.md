# Evaluative Reasoning LLM Benchmark

Dissertation research codebase: testing whether LLMs can replicate expert evaluative reasoning on UN evaluation report conclusions.

## Directory Structure

    eval-reasoning-benchmark/
    ├── config/              # API configuration, model parameters
    │   └── prompts/         # Zero-shot and few-shot prompt templates
    ├── src/
    │   ├── extraction/      # Phase 1: fragment extraction support
    │   ├── api_client/      # G.1-G.3: API calls, retry, parsing
    │   ├── pilot/            # Pre-run cost/token audit (see spec below)
    │   ├── scoring/         # G.4-G.6: coherence, accuracy, adjudication
    │   └── analysis/        # H-series: stats, visualization
    ├── data/
    │   ├── raw/             # Fragment corpus
    │   ├── gold_standard/   # Locked expert judgments
    │   └── responses/       # Model outputs
    ├── logs/                # Execution and session logs
    ├── appendices/          # Templates from E.1-E.4, D.1-D.3
    └── tests/               # Unit tests for scoring logic

## Environment Setup

This project requires **Python 3.10+** (Gemini 3.1 Pro Preview High's `google-genai>=2.3.0` dependency won't install on anything older — the other 5 models don't care, but the whole project is run as one codebase). macOS's default `python3` is often an older Anaconda/system install; check `python3 --version` before assuming it's new enough.

Set up a project-local virtual environment once, using whichever 3.10+ interpreter you have (find one via `ls /Library/Frameworks/Python.framework/Versions/` or `pyenv versions` if you use pyenv):

```
/path/to/your/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then, every time you work in this repo, activate it first:

```
source .venv/bin/activate
```

`python3`/`pip` now resolve to `.venv`'s copies for the rest of that terminal session — no need to type the full interpreter path each time. `.venv/` is gitignored; recreate it (same two commands) on any new machine rather than committing it.

## API Key Setup

The benchmark calls 6 model providers. Each needs its API key set as a plain environment variable before running — the codebase uses `os.getenv()` directly, with no `.env`-file loading (`.env` is gitignored but unused by convention).

| Env var | Provider | Model |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI (Responses API) | GPT-5.6 Sol, Max Effort |
| `ANTHROPIC_API_KEY` | Anthropic | Claude Opus 5, thinking x high effort |
| `GOOGLE_API_KEY` | Google | Gemini 3 Pro Preview High |
| `DEEPSEEK_API_KEY` | DeepSeek | DeepSeek V4 Pro |
| `MOONSHOT_API_KEY` | Moonshot AI | Kimi K3 |
| `FIREWORKS_API_KEY` | Fireworks | GLM 5.2 (served via Fireworks, not Zhipu/BigModel) |

See `config/api_config.py` for the authoritative model/provider mapping. Most `model_id`/`endpoint`/reasoning-effort values are now verified against provider docs; a few remaining `# verify at execution` markers and inline `STRUCTURAL FLAG` comments there call out shapes worth double-checking before a real run (notably GPT-5.6 Sol's Responses API response envelope and Kimi K3's exact endpoint path).

## Author

Zach Tilton — Interdisciplinary Ph.D. in Evaluation, Western Michigan University
