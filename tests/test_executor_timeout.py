"""
Unit tests for src/api_client/executor.call_with_hard_timeout.

Added after a real hung call during the first live benchmark session
(2026-10-20): execute_google_interactions_request's call to Google's
google-genai SDK had no timeout of its own and sat open for 40+ minutes
with no response, bypassing retry.make_api_call_with_retry's retry/backoff
entirely since that logic only runs once the call returns or raises.
call_with_hard_timeout is the fix -- these tests prove it actually cuts off
a hanging call rather than just reading correctly.
"""

from __future__ import annotations

import time

import pytest

from src.api_client.executor import call_with_hard_timeout


def _sleep_then_return(seconds: float, value: str = "done") -> str:
    time.sleep(seconds)
    return value


class TestCallWithHardTimeout:
    def test_fast_call_returns_normally(self):
        result = call_with_hard_timeout(_sleep_then_return, 2.0, 0.01)
        assert result == "done"

    def test_slow_call_raises_timeout_error_promptly(self):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            call_with_hard_timeout(_sleep_then_return, 0.2, 5.0)
        elapsed = time.monotonic() - start

        # Must cut off near the budget, not silently wait out the full
        # 5-second sleep -- this is the exact failure mode being fixed.
        assert elapsed < 1.0

    def test_timeout_error_message_contains_timeout_keyword(self):
        """
        retry.APIError.categorize() string-matches "timeout"/"timed out" in
        str(exception) to route this into the TIMEOUT (retriable) category
        -- the message shape matters, not just that it raises.
        """
        with pytest.raises(TimeoutError, match="timed out"):
            call_with_hard_timeout(_sleep_then_return, 0.1, 5.0)

    def test_passes_through_args_and_kwargs(self):
        result = call_with_hard_timeout(_sleep_then_return, 2.0, 0.01, value="custom")
        assert result == "custom"

    def test_propagates_exception_raised_by_fn(self):
        def _raises():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            call_with_hard_timeout(_raises, 2.0)
