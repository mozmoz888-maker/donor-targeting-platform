"""Service configuration: paths, EV parameters, retry policy. Environment-overridable."""

from __future__ import annotations

import os
from pathlib import Path

from donor_targeting.config import REPO_ROOT

ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"
DECISIONS_DIR = REPO_ROOT / "data" / "decisions"
MODEL_PATH = REPO_ROOT / "models" / "donation_model.joblib"

# ---- expected-value rule: email iff p_donate * MEAN_GIFT >= P_UNSUB * LTV + SEND_COST
# LTV default = measured 6-month expected value per subscriber (reports/cadence.json);
# raise it to suppress more (each $10 of LTV moves the p_donate cutoff by ~0.26 points)
MEAN_GIFT_USD = float(os.environ.get("MEAN_GIFT_USD", 57.94))
P_UNSUB = float(os.environ.get("P_UNSUB", 0.015))
LTV_USD = float(os.environ.get("LTV_USD", 19.62))
SEND_COST_USD = float(os.environ.get("SEND_COST_USD", 0.002))

# dormant actives (no 30-day engagement -> unknown taste) enter only via a rescue slice,
# deterministically per (request_id, person_id) so replays and idempotency hold
DORMANT_INCLUDE_PROB = float(os.environ.get("DORMANT_INCLUDE_PROB", 0.15))

# optional post-gift cooldown (emails to suppress after a gift); 0 = off
COOLDOWN_EMAILS = int(os.environ.get("COOLDOWN_EMAILS", 0))

# ---- CRM retry ladder under the 3-minute SLA --------------------------------
SLA_SECONDS = float(os.environ.get("SLA_SECONDS", 180.0))
RETRY_WAITS_S = [float(w) for w in os.environ.get("RETRY_WAITS_S", "10,20,40,80").split(",")]
EXPECTED_CRM_CALL_S = 2.0
DEADLINE_MARGIN_S = 15.0  # never start an attempt that can't finish with this much slack

# failure injection for the resilience demo (0 = never fail)
CRM_FAIL_FIRST = int(os.environ.get("SERVICE_CRM_FAIL_FIRST", 0))

EMBEDDING_DIM = 1024
PRECOMPUTE_CHUNK = 50_000
