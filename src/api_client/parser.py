"""
Response parsing, classification normalization, and rationale validation.

Covers Appendix G.3 parsing sections.  No I/O occurs here; all functions
are pure transformations of strings/dicts to support easy unit testing.
"""

from __future__ import annotations

import json
import re


def extract_response_content(response_json: dict, model: str) -> str:
    """
    Extract the text content field from a raw API response dict.

    Handles all three wire formats used in the benchmark:
    - OpenAI-compatible (GPT, DeepSeek, Kimi, GLM): ``choices[0].message.content``
    - Anthropic (Claude): ``content[0].text``
    - Google Gemini: ``candidates[0].content.parts[0].text``

    Args:
        response_json: Raw JSON-decoded response from the API.
        model: Model identifier (used in error messages only).

    Returns:
        Extracted text string.

    Raises:
        ValueError: If the response structure matches none of the known formats.
    """
    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"]

    if "content" in response_json:
        return response_json["content"][0]["text"]

    if "candidates" in response_json:
        return response_json["candidates"][0]["content"]["parts"][0]["text"]

    raise ValueError(
        f"Unrecognized API response format for model '{model}'. "
        f"Top-level keys present: {list(response_json.keys())}"
    )


def parse_structured_json(content: str) -> dict:
    """
    Parse a JSON-formatted model response.

    Expected payload::

        {"classification": "sound", "rationale": "2-4 sentence explanation"}

    Markdown code fences (````` ```json ... ``` `````) are stripped before
    parsing so models that wrap JSON in fences are handled correctly.

    Args:
        content: Raw text extracted from the API response.

    Returns:
        Dict with string keys ``classification`` and ``rationale``.

    Raises:
        json.JSONDecodeError: Content is not valid JSON after fence stripping.
        KeyError: Expected keys are absent from the parsed object.
    """
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        content = content.strip()

    parsed = json.loads(content)
    return {
        "classification": parsed["classification"].strip().lower(),
        "rationale": parsed["rationale"].strip(),
    }


def parse_unstructured_text(content: str) -> dict:
    """
    Parse a free-text model response using regex pattern matching.

    Handles labeled formats like::

        Classification: sound
        Rationale: The evaluation demonstrates...

    and unlabeled formats where the classification appears inline.

    "not sound" is searched before "sound" to prevent the "sound" pattern
    from falsely matching inside "not sound".

    Args:
        content: Raw text extracted from the API response.

    Returns:
        Dict with keys ``classification`` (str or None) and
        ``rationale`` (str or None).
    """
    # Labeled classification field
    class_match = re.search(
        r"(?:classification|judgment|verdict)\s*:\s*(not\s+sound|sound)",
        content,
        re.IGNORECASE,
    )
    if class_match:
        classification = class_match.group(1).strip().lower()
    else:
        # Unlabeled fallback — "not sound" searched before "sound"
        if re.search(r"\bnot\s+sound\b", content, re.IGNORECASE):
            classification = "not sound"
        elif re.search(r"\bsound\b", content, re.IGNORECASE):
            classification = "sound"
        else:
            classification = None

    # Labeled rationale field
    rationale_match = re.search(
        r"(?:rationale|explanation|reasoning)\s*:\s*(.+?)(?:\n{2,}|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    elif class_match:
        # Take everything after the labeled classification line
        after_class = content[class_match.end():].strip()
        rationale = after_class or None
    elif classification:
        # Take everything after the first occurrence of the bare classification
        pattern = r"\bnot\s+sound\b" if classification == "not sound" else r"\bsound\b"
        parts = re.split(pattern, content, maxsplit=1, flags=re.IGNORECASE)
        rationale = parts[1].strip() if len(parts) > 1 else None
    else:
        rationale = content.strip() or None

    return {"classification": classification, "rationale": rationale}


def get_input_tokens(response_json: dict, model: str) -> int:  # noqa: ARG001
    """
    Extract the prompt (input) token count from an API response.

    Handles:
    - OpenAI-compatible: ``usage.prompt_tokens``
    - Anthropic: ``usage.input_tokens``
    - Gemini: ``usageMetadata.promptTokenCount``

    Args:
        response_json: Raw JSON-decoded API response.
        model: Model identifier (unused; kept for signature symmetry).

    Returns:
        Input token count, or 0 if unavailable.
    """
    if "usage" in response_json:
        usage = response_json["usage"]
        return usage.get("prompt_tokens") or usage.get("input_tokens", 0)

    if "usageMetadata" in response_json:
        return response_json["usageMetadata"].get("promptTokenCount", 0)

    return 0


def get_output_tokens(response_json: dict, model: str) -> int:  # noqa: ARG001
    """
    Extract the completion (output) token count from an API response.

    Handles:
    - OpenAI-compatible: ``usage.completion_tokens``
    - Anthropic: ``usage.output_tokens``
    - Gemini: ``usageMetadata.candidatesTokenCount``

    Args:
        response_json: Raw JSON-decoded API response.
        model: Model identifier (unused; kept for signature symmetry).

    Returns:
        Output token count, or 0 if unavailable.
    """
    if "usage" in response_json:
        usage = response_json["usage"]
        return usage.get("completion_tokens") or usage.get("output_tokens", 0)

    if "usageMetadata" in response_json:
        return response_json["usageMetadata"].get("candidatesTokenCount", 0)

    return 0


def parse_api_response(response_json: dict, model: str) -> dict:
    """
    Parse a raw API response into a standardized intermediate dict.

    Attempts structured JSON parsing first; falls back to text pattern
    matching if the response is not JSON or is wrapped in prose.

    Args:
        response_json: Raw JSON-decoded response from the API call.
        model: Model identifier for format detection and error messages.

    Returns:
        Dict with keys:
        - ``classification`` (str or None)
        - ``rationale`` (str or None)
        - ``token_count_input`` (int)
        - ``token_count_output`` (int)
        - ``parse_method`` (``'json'`` or ``'text'``)
    """
    content = extract_response_content(response_json, model)
    input_tokens = get_input_tokens(response_json, model)
    output_tokens = get_output_tokens(response_json, model)

    try:
        parsed = parse_structured_json(content)
        return {
            "classification": parsed["classification"],
            "rationale": parsed["rationale"],
            "token_count_input": input_tokens,
            "token_count_output": output_tokens,
            "parse_method": "json",
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass  # fall through to text parsing

    parsed = parse_unstructured_text(content)
    return {
        "classification": parsed["classification"],
        "rationale": parsed["rationale"],
        "token_count_input": input_tokens,
        "token_count_output": output_tokens,
        "parse_method": "text",
    }


def normalize_classification(raw: str | None) -> str | None:
    """
    Normalize variant classification phrasings to the schema's canonical values.

    Returns ``'sound'``, ``'not_sound'``, or ``None`` (unrecognizable).

    Not-sound variants are tested *before* sound variants so that "not sound"
    is never accidentally matched by the "sound" substring check.

    Args:
        raw: Raw classification string from model output.

    Returns:
        ``'sound'``, ``'not_sound'``, or ``None``.
    """
    if not raw:
        return None

    normalized = raw.strip().lower()

    # Not-sound checked first — prevents "sound" substring match on "not sound"
    not_sound_variants = [
        "not sound",
        "unsound",
        "not valid",
        "invalid",
        "indefensible",
        "unacceptable",
        "fail",
        "weak",
        "no",
    ]
    if any(variant in normalized for variant in not_sound_variants):
        return "not_sound"

    # Sound variants — exact match to avoid false positives on longer strings
    sound_variants = {"sound", "valid", "acceptable", "defensible", "pass", "yes"}
    if normalized in sound_variants:
        return "sound"

    return None  # ambiguous; caller will set error_flag


def validate_rationale(rationale: str | None) -> tuple[bool, str | None]:
    """
    Validate that a rationale meets minimum quality criteria from the spec.

    Criteria (Edge Case Rules, spec):
    1. At least 10 words of content.
    2. Contains ≥2 domain-reasoning keywords (not a mere restatement).

    Args:
        rationale: Rationale text extracted from model output.

    Returns:
        Tuple of ``(is_valid: bool, issue_description: str | None)``.
        ``issue_description`` is ``None`` when valid.
    """
    if not rationale or not rationale.strip():
        return False, "Rationale is empty"

    words = rationale.split()
    if len(words) < 10:
        return False, f"Rationale too short ({len(words)} words; minimum 10 required)"

    reasoning_keywords = {
        "evidence", "criteria", "criterion", "standard", "warrant", "synthesis",
        "integration", "conclusion", "judgment", "evaluation", "reasoning",
        "argument", "basis", "support", "limitation", "qualification", "context",
        "demonstrates", "lacks", "shows", "addresses", "findings", "assessment",
        "analysis",
    }
    text_lower = rationale.lower()
    keyword_hits = sum(1 for kw in reasoning_keywords if kw in text_lower)

    if keyword_hits < 2:
        return False, "Rationale lacks substantive reasoning (fewer than 2 domain keywords)"

    return True, None
