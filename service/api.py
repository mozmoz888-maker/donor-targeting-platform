"""The production API.

POST /audiences        - the routing agent's request in, the mail list pushed to the CRM
GET  /audiences/{id}   - status/decision record for a request
GET  /healthz          - liveness + artifact freshness

Error policy (agreed):
* schema/dimension errors fail fast (422) - retrying a malformed request fixes nothing
* duplicate request_id is idempotent (409 + the original result, never two CRM lists)
* transient CRM errors retry 10/20/40/80s under the 180s SLA deadline, then the request
  terminates as failed with the computed audience persisted for a later re-push

    uv run uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from donor_targeting.config import PROGRAM_AREAS
from service import config, crm, scoring

app = FastAPI(title="donor targeting service", version="1.0.0")


class Content(BaseModel):
    content_id: int
    subject: str
    program_area: str
    dense_embedding: list[float] = Field(min_length=1024, max_length=1024)


class AudienceRequest(BaseModel):
    request_id: str
    org_id: str
    requested_at: str
    content: Content
    audience: dict
    destination: dict


def _record_path(request_id: str) -> Path:
    safe = "".join(c for c in request_id if c.isalnum() or c in "-_")
    d = config.DECISIONS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_record(request_id: str, record: dict) -> None:
    d = _record_path(request_id)
    (d / "decision.json").write_text(json.dumps(record, indent=2, default=str))
    with (config.DECISIONS_DIR / "index.jsonl").open("a") as ix:
        ix.write(json.dumps(record, default=str) + "\n")


def _read_record(request_id: str) -> dict | None:
    p = _record_path(request_id) / "decision.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.post("/audiences")
def create_audience(req: AudienceRequest):
    t0 = time.monotonic()

    if req.content.program_area not in PROGRAM_AREAS:
        raise HTTPException(422, f"unknown program_area {req.content.program_area!r}")
    emb = np.asarray(req.content.dense_embedding, dtype=np.float32)
    if not np.isfinite(emb).all():
        raise HTTPException(422, "dense_embedding contains non-finite values")

    if (prior := _read_record(req.request_id)) is not None:
        raise HTTPException(
            409,
            {"detail": "request_id already processed", "prior_status": prior["status"],
             "prior_list_id": prior.get("list_id")},
        )

    try:
        state = scoring.get_state()
        result = state.score(emb, req.request_id)
    except FileNotFoundError:
        raise HTTPException(503, "scoring artifacts missing: run make precompute")
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    ids = result["person_ids"][result["mailed"]]
    reasons = result["suppressed_reasons"]

    try:
        list_id, attempts, waits = crm.push_with_retry(
            crm.build_crm(),
            req.destination.get("list_name", req.request_id),
            ids.tolist(),
            req.content.content_id,
            deadline_s=config.SLA_SECONDS,
        )
    except crm.CRMExhausted as e:
        # graceful stop: terminal FAILED status, audience persisted for a later re-push
        np.savetxt(_record_path(req.request_id) / "mailed_ids.txt", ids, fmt="%d")
        record = {
            "request_id": req.request_id, "status": "failed", "reason": str(e),
            "attempts": getattr(e, "attempts", 0), "waits": getattr(e, "waits", []),
            "mailed_count": int(result["mailed"].sum()), "suppressed_reasons": reasons,
            "params": result["params"], "timings": result["timings"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "note": "audience persisted to mailed_ids.txt for manual/later re-push",
        }
        _write_record(req.request_id, record)
        raise HTTPException(503, record)

    elapsed = round(time.monotonic() - t0, 2)
    record = {
        "request_id": req.request_id, "status": "sent", "list_id": list_id,
        "mailed_count": int(result["mailed"].sum()),
        "suppressed_count": int((~result["mailed"]).sum()),
        "suppressed_reasons": reasons,
        "attempts": attempts, "retry_waits": waits,
        "params": result["params"],
        "timings": {**result["timings"], "total_s": elapsed},
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    np.savetxt(_record_path(req.request_id) / "mailed_ids.txt", ids, fmt="%d")
    _write_record(req.request_id, record)
    return record


@app.get("/audiences/{request_id}")
def audience_status(request_id: str):
    if (record := _read_record(request_id)) is None:
        raise HTTPException(404, "unknown request_id")
    return record


@app.get("/healthz")
def healthz():
    try:
        state = scoring.get_state()
        built = datetime.fromisoformat(state.meta["built_at"])
        age_h = (datetime.now(timezone.utc) - built).total_seconds() / 3600
        return {
            "ok": True,
            "active_people": state.meta["active_people"],
            "with_centroid": state.meta["with_centroid"],
            "artifacts_built_at": state.meta["built_at"],
            "artifact_age_hours": round(age_h, 1),
            "snapshot_at": state.meta["snapshot_at"],
            "staleness_ok": age_h < 32,  # nightly refresh + slack
        }
    except Exception as e:  # noqa: BLE001 - healthz must never 500 without a body
        return {"ok": False, "error": str(e)}
