"""
API error categorization, exponential backoff, retry logic, and rerun queue.

Covers Appendix G.2.  The retry schedule matches the spec exactly:
  attempt 1 → wait 10 s, attempt 2 → wait 30 s, attempt 3 → wait 90 s.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .config import FAILED_CALLS_LOG


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class APIError:
    """
    Error category constants and classification logic for API call failures.

    Categories drive retry decisions: transient errors are retried with
    backoff; permanent errors are logged and skipped immediately.
    """

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_RESPONSE = "invalid_response"
    SAFETY_TRIGGER = "safety_policy_triggered"
    API_ERROR = "api_error"
    OTHER = "other"

    # Transient errors that warrant automatic retry
    RETRIABLE: frozenset[str] = frozenset({TIMEOUT, RATE_LIMIT, SERVICE_UNAVAILABLE})
    # Permanent errors where retry is unlikely to succeed
    PERMANENT: frozenset[str] = frozenset({SAFETY_TRIGGER, API_ERROR, INVALID_RESPONSE})

    @staticmethod
    def categorize(
        error: Exception,
        response_text: str | None = None,
    ) -> tuple[str, str]:
        """
        Classify an exception into an error category and message pair.

        Checks the stringified exception for known HTTP status codes and
        keywords; also inspects ``response_text`` for safety-policy language.

        Args:
            error: Exception raised during the API call.
            response_text: Raw response body string, if available.

        Returns:
            Tuple of (category: str, message: str).
        """
        err = str(error).lower()

        if "timeout" in err or "timed out" in err:
            return APIError.TIMEOUT, str(error)

        if "rate limit" in err or "429" in err:
            return APIError.RATE_LIMIT, str(error)

        if any(tok in err for tok in ("503", "502", "service unavailable", "unavailable")):
            return APIError.SERVICE_UNAVAILABLE, str(error)

        if response_text and any(
            phrase in response_text.lower()
            for phrase in (
                "content policy",
                "safety policy",
                "harmful content",
                "policy violation",
            )
        ):
            return APIError.SAFETY_TRIGGER, "Safety policy triggered"

        if any(tok in err for tok in ("json", "parse", "decode", "format")):
            return APIError.INVALID_RESPONSE, str(error)

        if any(tok in err for tok in ("400", "401", "403", "api error")):
            return APIError.API_ERROR, str(error)

        return APIError.OTHER, str(error)


# ---------------------------------------------------------------------------
# Backoff helpers
# ---------------------------------------------------------------------------

def exponential_backoff(attempt: int) -> int:
    """
    Return the wait time in seconds for a given retry attempt.

    Schedule (spec Section 2.3):
      - attempt 1 → 10 s
      - attempt 2 → 30 s
      - attempt 3 → 90 s

    Args:
        attempt: 1-based retry attempt number (the attempt that just failed).

    Returns:
        Seconds to wait before the next attempt.
    """
    schedule = {1: 10, 2: 30, 3: 90}
    return schedule.get(attempt, 90)


def wait_with_progress(seconds: int, label: str = "Waiting") -> None:
    """
    Sleep for ``seconds`` with a printed progress indicator.

    Args:
        seconds: Duration to sleep.
        label: Prefix text for the printed message.
    """
    print(f"  {label} {seconds}s...", end="", flush=True)
    time.sleep(seconds)
    print(" Done.")


def should_retry(category: str, attempt: int, max_attempts: int = 3) -> bool:
    """
    Decide whether an API call should be retried.

    Args:
        category: Error category from :meth:`APIError.categorize`.
        attempt: The 1-based attempt number that just failed.
        max_attempts: Total attempts allowed (initial + retries).

    Returns:
        ``True`` if the call should be retried.
    """
    if attempt >= max_attempts:
        return False

    if category in APIError.RETRIABLE:
        return True

    if category in APIError.PERMANENT:
        return False

    # Unknown error: retry once only
    return attempt == 1


# ---------------------------------------------------------------------------
# Main retry wrapper
# ---------------------------------------------------------------------------

def make_api_call_with_retry(
    fragment: dict,
    model: str,
    prompt_type: str,
    run_number: int,
    max_attempts: int = 3,
) -> dict:
    """
    Execute an API call with automatic retry on transient errors.

    Each attempt is a fully independent, stateless call with no conversation
    history or accumulated context (spec Section 2.3).  Uses the backoff
    schedule 10 s / 30 s / 90 s between attempts.

    Args:
        fragment: Dict with keys ``'id'`` (str) and ``'text'`` (str).
        model: Model identifier from ``API_CONFIG``.
        prompt_type: ``'zero_shot'`` or ``'few_shot'``.
        run_number: Run number 1–3.
        max_attempts: Total attempts allowed (initial call + retries).

    Returns:
        Dict with keys:
        - ``fragment_id``, ``model``, ``prompt_type``, ``run_number``
        - ``status``: ``'success'`` or ``'failed'``
        - ``parsed_response``: output from :func:`executor.execute_api_request`,
          or ``None`` on failure
        - ``attempts``: number of attempts made
        - ``timestamp``: ISO-format string
        - ``error``: ``None`` on success, or
          ``{'category': str, 'message': str}`` on failure
    """
    # Deferred import to avoid circular dependency at module load time
    from .executor import execute_api_request  # noqa: PLC0415

    last_error: tuple[str, str] | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            parsed = execute_api_request(fragment, model, prompt_type, run_number)
            return {
                "fragment_id": fragment["id"],
                "model": model,
                "prompt_type": prompt_type,
                "run_number": run_number,
                "status": "success",
                "parsed_response": parsed,
                "attempts": attempt,
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }

        except Exception as exc:
            # Attempt to retrieve the raw response body for safety-trigger detection
            try:
                response_text = exc.response.text if hasattr(exc, "response") else None
            except Exception:
                response_text = None

            category, message = APIError.categorize(exc, response_text)
            last_error = (category, message)
            print(
                f"  Attempt {attempt}/{max_attempts} failed "
                f"[{category}]: {message[:120]}"
            )

            if should_retry(category, attempt, max_attempts):
                wait_with_progress(exponential_backoff(attempt), label="Retrying in")
            else:
                break  # permanent or unknown error — skip remaining attempts

    category, message = last_error or (APIError.OTHER, "Unknown error")
    return {
        "fragment_id": fragment["id"],
        "model": model,
        "prompt_type": prompt_type,
        "run_number": run_number,
        "status": "failed",
        "parsed_response": None,
        "attempts": attempt,
        "timestamp": datetime.now().isoformat(),
        "error": {"category": category, "message": message},
    }


# ---------------------------------------------------------------------------
# Failed-call queue management (spec Section 2.3)
# ---------------------------------------------------------------------------

def log_failed_call(
    result: dict,
    log_path: Path = FAILED_CALLS_LOG,
) -> None:
    """
    Append a failed API call record to the JSONL failure log.

    Records are appended (not overwritten) so the log accumulates across
    sessions.  Callers use :func:`load_failed_calls` + :func:`rerun_failed_calls`
    to re-attempt logged failures in a separate session.

    Args:
        result: Result dict from :func:`make_api_call_with_retry` with
                ``status == 'failed'``.
        log_path: Path to the JSONL log file.
    """
    if result["status"] != "failed":
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "fragment_id": result["fragment_id"],
        "model": result["model"],
        "prompt_type": result["prompt_type"],
        "run_number": result["run_number"],
        "error_category": result["error"]["category"],
        "error_message": result["error"]["message"],
        "attempts": result["attempts"],
        "timestamp": result["timestamp"],
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(
        f"  Logged failed call: {result['fragment_id']} / {result['model']} "
        f"/ {result['prompt_type']} / run {result['run_number']}"
    )


def load_failed_calls(log_path: Path = FAILED_CALLS_LOG) -> list[dict]:
    """
    Load failed call records from the JSONL log for a rerun session.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        List of failed-call record dicts (empty list if file not found).
    """
    if not log_path.exists():
        print(f"No failed calls log found at {log_path}")
        return []

    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} failed calls for rerun.")
    return records


def rerun_failed_calls(
    failed_calls: list[dict],
    fragment_data: dict,
) -> list[dict]:
    """
    Retry every call from the failed-call log in a dedicated rerun session.

    Each call is re-attempted via :func:`make_api_call_with_retry` with the
    full retry schedule.  A 5-second pause separates reruns.

    Args:
        failed_calls: List of failed-call dicts from :func:`load_failed_calls`.
        fragment_data: Dict mapping ``fragment_id`` → fragment dict
                       (with ``'id'`` and ``'text'`` keys).

    Returns:
        List of result dicts from :func:`make_api_call_with_retry`.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"RERUN SESSION — {len(failed_calls)} calls to retry")
    print(f"{sep}\n")

    results: list[dict] = []

    for i, call in enumerate(failed_calls, start=1):
        fragment = fragment_data.get(call["fragment_id"])
        if fragment is None:
            print(
                f"[{i}/{len(failed_calls)}] ERROR: fragment "
                f"'{call['fragment_id']}' not found — skipping"
            )
            continue

        print(
            f"[{i}/{len(failed_calls)}] {call['fragment_id']} / {call['model']} "
            f"/ {call['prompt_type']} / run {call['run_number']}"
        )
        result = make_api_call_with_retry(
            fragment=fragment,
            model=call["model"],
            prompt_type=call["prompt_type"],
            run_number=call["run_number"],
        )
        results.append(result)

        if i < len(failed_calls):
            time.sleep(5)

    successes = sum(1 for r in results if r["status"] == "success")
    still_failed = len(results) - successes
    print(f"\n{sep}")
    print(f"RERUN SUMMARY: {successes}/{len(results)} succeeded, {still_failed} still failed")
    print(f"{sep}\n")

    return results


def clear_failed_calls_log(log_path: Path = FAILED_CALLS_LOG) -> None:
    """
    Delete the failed-calls JSONL log after a successful rerun session.

    Args:
        log_path: Path to the JSONL log file.
    """
    if log_path.exists():
        log_path.unlink()
        print(f"Cleared failed calls log: {log_path}")
    else:
        print(f"No failed calls log to clear at {log_path}")
