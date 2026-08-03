"""
Request construction, API call execution, and response database persistence.

Covers Appendix G.3 request-building and storage sections.

Design notes:
- G.3 contained two duplicate definitions of build_request_headers and
  build_request_payload; the more complete parameterized versions are kept.
- save_response_record appends one CSV row at a time (not full re-write)
  so that each response is persisted immediately per spec Section 2.3.
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

from .config import (
    ANTHROPIC_API_VERSION,
    API_CONFIG,
    MODEL_API_FAMILY,
    PARAM_MAPPING,
    PROMPTS_DIR,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSE_DB_PATH,
    STANDARD_PARAMS,
)
from .parser import (
    normalize_classification,
    parse_api_response,
    parse_structured_json,
    parse_unstructured_text,
    validate_rationale,
)

# ---------------------------------------------------------------------------
# Response database schema
# ---------------------------------------------------------------------------

# Column order matches the response_database schema in specs Section Data Schemas.
RESPONSE_DB_COLUMNS: list[str] = [
    "response_id",
    "fragment_id",
    "model_family",
    "prompt_condition",
    "run_number",
    "classification_output",
    "rationale_text",
    "timestamp",
    "api_latency_seconds",
    "token_count_input",
    "token_count_output",
    "api_version",
    "parse_method",
    "error_flag",
    "error_details",
]


def generate_response_id() -> str:
    """
    Generate a unique primary key for a response record.

    Returns:
        UUID4 string (e.g. ``'3f2504e0-4f89-11d3-9a0c-0305e82c3301'``).
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def filter_params(model: str, universal_params: dict) -> dict:
    """
    Translate universal parameter names to provider-specific names.

    Parameters whose provider-specific name is ``None`` are omitted (not
    supported by that API family). Parameters listed in the model's own
    ``unsupported_params`` (API_CONFIG) are dropped first — some models are
    stricter than their family default (e.g. Claude Opus 5 rejects all
    sampling params, unlike older Anthropic models under the same family).

    Args:
        model: Model identifier from ``API_CONFIG``.
        universal_params: Dict of universal parameter names → values.

    Returns:
        Dict with provider-specific parameter names, unsupported params dropped.
    """
    model_config = API_CONFIG.get(model, {})
    excluded = model_config.get("unsupported_params", frozenset())
    universal_params = {
        name: value for name, value in universal_params.items() if name not in excluded
    }

    api_family = MODEL_API_FAMILY.get(model, "openai_compatible")
    mapping = PARAM_MAPPING[api_family]
    return {
        provider_name: value
        for universal_name, value in universal_params.items()
        if (provider_name := mapping.get(universal_name)) is not None
    }


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def build_request_headers(config: dict) -> dict:
    """
    Construct HTTP authentication headers for an API call.

    Args:
        config: Model config dict from ``API_CONFIG``.

    Returns:
        Dict of HTTP header name → value pairs.

    Raises:
        ValueError: If the required API key environment variable is unset,
                    or if ``auth_type`` is not recognized.
    """
    env_var = config["api_key_env"]
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"API key not found. Set the '{env_var}' environment variable "
            f"before running the benchmark."
        )

    auth_type = config["auth_type"]

    if auth_type == "bearer":
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    if auth_type == "x-api-key":
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

    if auth_type == "api_key_param":
        # Gemini authenticates via URL query parameter, not a header
        return {"Content-Type": "application/json"}

    raise ValueError(
        f"Unknown auth_type '{auth_type}' in model config for "
        f"'{config.get('model_id', 'unknown')}'."
    )


def build_endpoint_url(config: dict) -> str:
    """
    Return the full endpoint URL, appending the API key for Gemini.

    Args:
        config: Model config dict from ``API_CONFIG``.

    Returns:
        Full URL string ready for ``requests.post()``.

    Raises:
        ValueError: If Gemini's API key environment variable is unset.
    """
    endpoint = config["endpoint"]

    if config["auth_type"] == "api_key_param":
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(
                f"API key not found. Set '{config['api_key_env']}' "
                "environment variable."
            )
        return f"{endpoint}?key={api_key}"

    return endpoint


def build_request_payload(config: dict, prompt: str, model: str) -> dict:
    """
    Construct the JSON request body with model-specific formatting.

    Applies parameter filtering via :func:`filter_params` so that
    unsupported parameters (e.g. ``frequency_penalty`` for Claude) are
    automatically excluded.

    Args:
        config: Model config dict from ``API_CONFIG``.
        prompt: Fully rendered prompt string.
        model: Model identifier (used to determine API family and filtering).

    Returns:
        Dict suitable for the ``json=`` argument of ``requests.post()``.
    """
    params = filter_params(model, STANDARD_PARAMS)
    api_family = MODEL_API_FAMILY.get(model, "openai_compatible")
    reasoning_payload = config.get("reasoning_effort_payload", {})

    if api_family == "anthropic":
        # Anthropic requires max_tokens at the top level, not inside a params block
        max_tokens = params.pop("max_tokens", STANDARD_PARAMS["max_tokens"])
        payload: dict = {
            "model": config["model_id"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        payload.update(params)
        payload.update(reasoning_payload)
        return payload

    if api_family == "google":
        generation_config = dict(params)
        generation_config.update(reasoning_payload)
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

    if api_family == "openai_responses":
        # OpenAI's Responses API (GPT-5.6 Sol) — distinct from Chat
        # Completions: single "input" string instead of a "messages" array.
        # See the STRUCTURAL FLAG comment on gpt_5_6_sol in
        # config/api_config.py for what else this family needed.
        payload = {
            "model": config["model_id"],
            "input": prompt,
        }
        payload.update(params)
        payload.update(reasoning_payload)
        return payload

    # OpenAI-compatible Chat Completions (DeepSeek, Kimi, GLM via Fireworks)
    payload = {
        "model": config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(params)
    payload.update(reasoning_payload)
    return payload


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt_template(
    prompt_type: str,
    fragment_text: str,
    prompts_dir: Path = PROMPTS_DIR,
) -> str:
    """
    Load a prompt template file and render it with the fragment text.

    Template files must contain the exact placeholder string
    ``[Fragment text will be inserted here]``.

    Args:
        prompt_type: ``'zero_shot'`` or ``'few_shot'``.
        fragment_text: Raw text of the evaluation fragment to assess.
        prompts_dir: Directory containing the ``*_template.txt`` files.

    Returns:
        Fully rendered prompt string ready for the API.

    Raises:
        FileNotFoundError: Template file does not exist.
        ValueError: Placeholder string is absent from the template.
    """
    template_path = prompts_dir / f"{prompt_type}_template.txt"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}\n"
            "Ensure config/prompts/ contains zero_shot_template.txt "
            "and few_shot_template.txt."
        )

    template = template_path.read_text(encoding="utf-8")
    placeholder = "[Fragment text will be inserted here]"
    if placeholder not in template:
        raise ValueError(
            f"Placeholder '{placeholder}' not found in template: {template_path}"
        )

    return template.replace(placeholder, fragment_text)


# ---------------------------------------------------------------------------
# API call execution
# ---------------------------------------------------------------------------

def execute_api_request(
    fragment: dict,
    model: str,
    prompt_type: str,
    run_number: int,  # noqa: ARG001  (passed for logging context by callers)
) -> dict:
    """
    Execute a single, fully stateless API call and return parsed data.

    No conversation history, memory, or accumulated context is carried
    between calls (spec Section 2.3).

    Args:
        fragment: Dict with keys ``'id'`` (str) and ``'text'`` (str).
        model: Model identifier from ``API_CONFIG``.
        prompt_type: ``'zero_shot'`` or ``'few_shot'``.
        run_number: Run number 1–3 (informational; not sent to the API).

    Returns:
        Dict with keys: ``classification``, ``rationale``,
        ``token_count_input``, ``token_count_output``, ``parse_method``,
        ``latency_seconds``, ``api_version``.

    Raises:
        requests.HTTPError: On non-2xx HTTP status (triggers retry in caller).
        requests.Timeout: On request timeout (triggers retry in caller).
        ValueError: On missing API key or configuration errors.
    """
    config = API_CONFIG[model]
    prompt = load_prompt_template(prompt_type, fragment["text"])

    if MODEL_API_FAMILY.get(model) == "google_interactions_sdk":
        # Gemini 3.1 Pro Preview High — Google's Interactions API, called
        # via the google-genai SDK rather than raw requests.post(). See the
        # STRUCTURAL FLAG comment on gemini_3_pro_preview_high in
        # config/api_config.py.
        return execute_google_interactions_request(config, prompt, model)

    headers = build_request_headers(config)
    payload = build_request_payload(config, prompt, model)
    endpoint = build_endpoint_url(config)

    start = time.monotonic()
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    latency = round(time.monotonic() - start, 3)

    response.raise_for_status()  # raises HTTPError for 4xx/5xx

    parsed = parse_api_response(response.json(), model)
    parsed["latency_seconds"] = latency
    parsed["api_version"] = config["model_id"]

    return parsed


def execute_google_interactions_request(config: dict, prompt: str, model: str) -> dict:
    """
    Execute a request against Google's Interactions API (google-genai SDK)
    for Gemini 3.1 Pro Preview High.

    Bypasses build_request_headers/build_request_payload/build_endpoint_url/
    requests.post entirely — the SDK owns request construction and auth.
    Requires google-genai>=2.3.0 (Python 3.10+; see requirements.txt).

    Field names used here (output_text; usage.total_input_tokens/
    total_output_tokens/total_thought_tokens) were confirmed by directly
    introspecting the installed google-genai 2.16.0 SDK's pydantic type
    definitions (Interaction.model_fields, Usage.model_fields) — real
    source, not a guess, but not exercised against a live API response
    either. Verify against one real call before trusting this at scale.

    Args:
        config: Model config dict from ``API_CONFIG`` (gemini_3_pro_preview_high).
        prompt: Fully rendered prompt string.
        model: Model identifier (used for filter_params' unsupported_params lookup).

    Returns:
        Dict with the same keys as the requests-based path: ``classification``,
        ``rationale``, ``token_count_input``, ``token_count_output``,
        ``reasoning_tokens``, ``parse_method``, ``latency_seconds``,
        ``api_version``.

    Raises:
        ValueError: On missing API key.
        Exception: Whatever google-genai raises for API errors — caught and
            categorized by retry.make_api_call_with_retry's generic
            exception handling (it already guards on ``hasattr(exc, "response")``
            rather than assuming a ``requests``-shaped exception).
    """
    from google import genai  # deferred import — only this one model needs the SDK

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"API key not found. Set the '{config['api_key_env']}' environment "
            f"variable before running the benchmark."
        )

    client = genai.Client(api_key=api_key)

    generation_config = filter_params(model, STANDARD_PARAMS)
    generation_config.update(config.get("reasoning_effort_payload", {}))

    start = time.monotonic()
    interaction = client.interactions.create(
        model=f"models/{config['model_id']}",
        input=prompt,
        generation_config=generation_config,
    )
    latency = round(time.monotonic() - start, 3)

    if interaction.status != "completed":
        # CONFIRMED via live diagnostic (2026-08-03): status is "incomplete"
        # when max_output_tokens is exhausted by thinking tokens before any
        # answer text is produced (same failure mode as DeepSeek V4 Pro's
        # max_tokens=500 issue — see model_params.py). Raising here surfaces
        # truncated responses as retryable errors instead of silently
        # persisting partial/cut-off rationale text.
        raise RuntimeError(
            f"Interactions API response had status={interaction.status!r} "
            f"(expected 'completed') — likely truncated by max_output_tokens. "
            f"Full interaction object: {interaction!r}"
        )

    text = interaction.output_text
    if not text:
        raise RuntimeError(
            "Interactions API response had no output_text. Full interaction "
            f"object: {interaction!r}"
        )

    usage = interaction.usage
    input_tokens = getattr(usage, "total_input_tokens", None) or 0
    output_tokens = getattr(usage, "total_output_tokens", None) or 0
    reasoning_tokens = getattr(usage, "total_thought_tokens", None) if usage else None

    try:
        parsed_text = parse_structured_json(text)
        classification, rationale, parse_method = (
            parsed_text["classification"], parsed_text["rationale"], "json",
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        parsed_text = parse_unstructured_text(text)
        classification, rationale, parse_method = (
            parsed_text["classification"], parsed_text["rationale"], "text",
        )

    return {
        "classification": classification,
        "rationale": rationale,
        "token_count_input": input_tokens,
        "token_count_output": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "parse_method": parse_method,
        "latency_seconds": latency,
        "api_version": config["model_id"],
    }


# ---------------------------------------------------------------------------
# Response database persistence
# ---------------------------------------------------------------------------

def save_response_record(
    record: dict,
    db_path: Path = RESPONSE_DB_PATH,
) -> None:
    """
    Append a single response record to the CSV database immediately.

    Creates the file with a header row on first write.  Per spec Section 2.3,
    each response is persisted as soon as it is received so that a session
    interruption loses at most one in-flight call.

    Args:
        record: Dict whose keys are a subset of ``RESPONSE_DB_COLUMNS``.
        db_path: Path to the response database CSV file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = db_path.exists() and db_path.stat().st_size > 0

    with db_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=RESPONSE_DB_COLUMNS,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def process_and_store_response(
    fragment: dict,
    model: str,
    prompt_type: str,
    run_number: int,
    parsed_response: dict,
    db_path: Path = RESPONSE_DB_PATH,
) -> dict:
    """
    Normalize, validate, assemble, and immediately persist a response record.

    Generates a unique ``response_id``, normalizes the classification to the
    schema's canonical values, validates the rationale, sets ``error_flag``
    and ``error_details`` accordingly, then calls :func:`save_response_record`
    to append the row to the CSV database.

    Args:
        fragment: Dict with ``'id'`` and ``'text'``.
        model: Model identifier from ``API_CONFIG``.
        prompt_type: ``'zero_shot'`` or ``'few_shot'``.
        run_number: Run number 1–3.
        parsed_response: Dict from :func:`execute_api_request`.
        db_path: Path to the response database CSV.

    Returns:
        The assembled record dict (same content written to CSV).
    """
    raw_classification = parsed_response.get("classification")
    rationale = parsed_response.get("rationale") or ""

    normalized_class = normalize_classification(raw_classification)
    is_valid, validation_issue = validate_rationale(rationale)

    error_flag = (normalized_class is None) or (not is_valid)
    if not is_valid:
        error_details: str | None = validation_issue
    elif normalized_class is None:
        error_details = f"Unrecognized classification: '{raw_classification}'"
    else:
        error_details = None

    record: dict = {
        "response_id": generate_response_id(),
        "fragment_id": fragment["id"],
        "model_family": model,
        "prompt_condition": prompt_type,
        "run_number": run_number,
        "classification_output": normalized_class,
        "rationale_text": rationale,
        "timestamp": datetime.now().isoformat(),
        "api_latency_seconds": parsed_response.get("latency_seconds"),
        "token_count_input": parsed_response.get("token_count_input", 0),
        "token_count_output": parsed_response.get("token_count_output", 0),
        "api_version": parsed_response.get("api_version"),
        "parse_method": parsed_response.get("parse_method"),
        "error_flag": error_flag,
        "error_details": error_details,
    }

    save_response_record(record, db_path)
    return record
