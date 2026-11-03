"""CRM client: the mock CRM plus a transient-failure wrapper and the retry ladder.

The production CRM takes the whole list in one request (~2 s). Transient failures retry
with exponential backoff (10/20/40/80 s) governed by the 3-minute SLA deadline: an attempt
is only started if it can still finish inside the deadline with margin. When the ladder is
exhausted the caller receives CRMExhausted and fails the request gracefully.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from donor_targeting.mocks.crm import MockCRM

from service import config


class TransientCRMError(RuntimeError):
    """A CRM failure that may resolve on retry (timeout / 5xx)."""


class CRMExhausted(RuntimeError):
    """Retries exhausted or deadline reached; the request must fail gracefully."""


class FlakyCRM:
    """Wraps the mock CRM; fails the first `fail_first` calls, then delegates.

    Used to demo/verify the retry path with evidence (SERVICE_CRM_FAIL_FIRST env).
    """

    def __init__(self, inner: MockCRM, fail_first: int) -> None:
        self.inner = inner
        self.remaining = fail_first

    def create_list(self, name: str, person_ids: Iterable[int], content_id: int | None = None) -> str:
        if self.remaining > 0:
            self.remaining -= 1
            raise TransientCRMError(f"injected transient CRM failure ({self.remaining} left)")
        return self.inner.create_list(name, person_ids, content_id=content_id)

    def get_list(self, list_id: str) -> dict:
        return self.inner.get_list(list_id)


def build_crm() -> FlakyCRM:
    return FlakyCRM(MockCRM(), config.CRM_FAIL_FIRST)


def push_with_retry(crm, name: str, person_ids, content_id: int | None, deadline_s: float):
    """Returns (list_id, attempts, waits_used). Raises CRMExhausted on ladder exhaustion."""
    t0 = time.monotonic()
    attempts = 0
    waits: list[float] = []
    while True:
        attempts += 1
        try:
            list_id = crm.create_list(name, person_ids, content_id=content_id)
            return list_id, attempts, waits
        except TransientCRMError:
            err = None
            if attempts > len(config.RETRY_WAITS_S):
                err = CRMExhausted(
                    f"CRM failed after {attempts - 1} retries (ladder {config.RETRY_WAITS_S})"
                )
            else:
                wait = config.RETRY_WAITS_S[attempts - 1]
                elapsed = time.monotonic() - t0
                if (elapsed + wait + config.EXPECTED_CRM_CALL_S + config.DEADLINE_MARGIN_S
                        > deadline_s):
                    err = CRMExhausted(
                        f"deadline governor: next retry needs {wait:.0f}s but only "
                        f"{deadline_s - elapsed:.0f}s of SLA remains"
                    )
                else:
                    time.sleep(wait)
                    waits.append(wait)
                    continue
            err.attempts = attempts
            err.waits = waits
            raise err
