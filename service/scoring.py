"""Scoring engine: artifacts in, mail/suppress decision out.

Loads the memory-mapped centroid matrix (float16, ~0.24 GB on disk for 244k x 1024) and the
fitted donation model once per worker; scoring a request is chunked dot products plus an
EV threshold. Nothing here touches the warehouse.
"""

from __future__ import annotations

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from service import config


class ScoringState:
    """Process-wide artifacts. Memory-mapped: RSS stays close to the touched chunks."""

    def __init__(self) -> None:
        meta = json.loads((config.ARTIFACTS_DIR / "meta.json").read_text())
        self.meta = meta
        digest = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
        if digest != meta.get("model_sha256"):
            raise RuntimeError(
                "model artifact does not match the hash recorded at precompute time; "
                "rerun make train + make precompute"
            )
        self.person_ids = pd.read_parquet(
            config.ARTIFACTS_DIR / "persons.parquet", columns=["person_id", "has_centroid",
                                                               "emails_since_gift"]
        )
        self.centroids = np.load(config.ARTIFACTS_DIR / "centroids.f16.npy", mmap_mode="r")
        self.norms = np.load(config.ARTIFACTS_DIR / "norms.f32.npy")
        self.model = joblib.load(config.MODEL_PATH)  # hash-verified above: our own artifact

    def _p_donate(self, sims: np.ndarray, has: np.ndarray) -> np.ndarray:
        # one feature, NaN for no-recent-engagement: the pipeline's imputer fills the
        # median and its add_indicator=True adds the was-null column the model was fit with
        X = np.where(has, sims, np.nan).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    def score(self, embedding: np.ndarray, request_id: str) -> dict:
        """Returns person_ids, mailed mask, p_donate, and per-reason suppress counts."""
        t0 = time.perf_counter()
        v = np.asarray(embedding, dtype=np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-12)

        n = len(self.person_ids)
        sims = np.zeros(n, dtype=np.float32)
        has = self.person_ids["has_centroid"].to_numpy()
        for s in range(0, n, config.PRECOMPUTE_CHUNK):
            e = min(s + config.PRECOMPUTE_CHUNK, n)
            block = np.asarray(self.centroids[s:e], dtype=np.float32)
            sims[s:e] = (block @ v) / np.maximum(self.norms[s:e], 1e-12)
        sims[~has] = np.nan  # no recent engagement: imputer + indicator handle it
        t_score = time.perf_counter() - t0

        p = self._p_donate(sims, has)
        ev_ok = p * config.MEAN_GIFT_USD >= config.P_UNSUB * config.LTV_USD + config.SEND_COST_USD

        mailed = ev_ok & has  # unknown taste never enters via EV, only via the rescue slice
        reasons = {
            "below_ev_threshold": int((~ev_ok & has).sum()),
            "dormant_no_recent_engagement": int((~has).sum()),
        }

        # dormant rescue: deterministic per (request_id, person) so replays are stable
        dormant_idx = np.flatnonzero(~has)
        if len(dormant_idx) and config.DORMANT_INCLUDE_PROB > 0:
            salt = f"{request_id}:dormant".encode()
            hashes = np.fromiter(
                (int.from_bytes(hashlib.sha256(salt + int(p).to_bytes(8, "little")).digest()[:8],
                                "little") / 2**64 for p in self.person_ids["person_id"]
                 .to_numpy()[dormant_idx]),
                dtype=np.float64, count=len(dormant_idx),
            )
            rescued = dormant_idx[hashes < config.DORMANT_INCLUDE_PROB]
            mailed[rescued] = True
            reasons["dormant_rescued"] = int(len(rescued))
            reasons["dormant_no_recent_engagement"] -= int(len(rescued))
        else:
            reasons["dormant_rescued"] = 0

        if config.COOLDOWN_EMAILS > 0:
            sg = self.person_ids["emails_since_gift"].to_numpy()
            cooled = mailed & (sg >= 0) & (sg < config.COOLDOWN_EMAILS)
            mailed &= ~cooled
            reasons["post_gift_cooldown"] = int(cooled.sum())

        return {
            "person_ids": self.person_ids["person_id"].to_numpy(),
            "mailed": mailed,
            "p_donate": p,
            "suppressed_reasons": reasons,
            "timings": {"score_s": round(t_score, 3)},
            "params": {
                "ltv_usd": config.LTV_USD, "send_cost_usd": config.SEND_COST_USD,
                "mean_gift_usd": config.MEAN_GIFT_USD, "p_unsub": config.P_UNSUB,
                "dormant_include_prob": config.DORMANT_INCLUDE_PROB,
                "threshold_p_donate": (config.P_UNSUB * config.LTV_USD + config.SEND_COST_USD)
                / config.MEAN_GIFT_USD,
            },
        }


_state: ScoringState | None = None


def get_state() -> ScoringState:
    global _state
    if _state is None:
        _state = ScoringState()
    return _state
