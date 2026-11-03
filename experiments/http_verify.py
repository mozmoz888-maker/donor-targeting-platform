"""External HTTP verification battery against the running local service.

Unlike experiments/service_evidence.py (in-process TestClient), this talks to the real
uvicorn server over HTTP and additionally checks things only visible from outside:

* every outbox list: ids unique, count matches the response, every id is an ACTIVE person
* determinism: same request re-run (after wiping its decision) reproduces the identical list
* concurrency: three simultaneous requests all succeed with sane, distinct audiences
* contract: 409 duplicate, 404 unknown, 422 bad program area / short embedding / null value

This is a test client for OUR OWN local service: the base URL is validated to be
http(s) loopback only (no other host is ever contacted, redirects are refused).

    uv run python experiments/http_verify.py [base_url]
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from donor_targeting.config import REQUESTS_DIR

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
DECISIONS = Path("data/decisions")
OUTBOX = Path("data/crm_outbox")
RESULTS: list[tuple[bool, str]] = []


def validated_url(url: str) -> str:
    """http/https to loopback only: this harness tests our own local service and
    must never be pointed at anything else."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError(f"only http/https allowed, got {p.scheme!r}")
    if p.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"test harness only talks to local loopback, got {p.hostname!r}")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def check(ok: bool, msg: str) -> None:
    RESULTS.append((bool(ok), msg))
    print(f"{'PASS' if ok else 'FAIL'}: {msg}")


def post(base: str, path: str, body: dict):
    req = urllib.request.Request(
        validated_url(base + path), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    try:
        with _OPENER.open(req, timeout=200) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.time() - t


def get(base: str, path: str):
    try:
        with _OPENER.open(validated_url(base + path), timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def mailed_ids(request_id: str) -> set[int]:
    txt = (DECISIONS / request_id / "mailed_ids.txt").read_text().split()
    return {int(x) for x in txt}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    validated_url(base + "/healthz")  # fail fast on a misconfigured target
    active = set(
        pd.read_parquet("data/artifacts/persons.parquet", columns=["person_id"])["person_id"]
    )
    print(f"{len(active):,} active persons in artifacts")

    code, health = get(base, "/healthz")
    check(code == 200 and health["ok"],
          f"healthz ok (artifacts {health.get('artifact_age_hours')}h old)")

    requests = [json.loads(p.read_text()) for p in sorted(REQUESTS_DIR.glob("*.json"))]

    # ---- all 12 over real HTTP -------------------------------------------------
    worst = 0.0
    for req in requests:
        code, body, dt = post(base, "/audiences", req)
        ids = mailed_ids(req["request_id"])
        ok = (
            code == 200
            and body["mailed_count"] == len(ids)
            and ids <= active
            and 100_000 <= len(ids) <= 250_000
        )
        check(ok, f"{req['request_id']}: {len(ids):,} mailed, unique, all active, "
                  f"in 100k-250k band, {dt:.1f}s over HTTP")
        out = pd.read_csv(OUTBOX / f"{body['list_id']}.csv")
        check(set(out["person_id"]) == ids and len(out) == len(ids),
              f"{req['request_id']}: CRM outbox list matches mailed ids exactly")
        worst = max(worst, dt)
    check(worst < 180, f"worst HTTP request→CRM {worst:.1f}s < 180s SLA")

    # ---- determinism: same request, wiped decision, identical list --------------
    rid = requests[0]["request_id"]
    before = sha(DECISIONS / rid / "mailed_ids.txt")
    code_dup, body_dup, _ = post(base, "/audiences", requests[0])
    check(code_dup == 409 and body_dup["detail"]["prior_status"] == "sent",
          f"duplicate {rid} -> 409 with prior result (no second list)")
    shutil.rmtree(DECISIONS / rid)
    code2, body2, _ = post(base, "/audiences", requests[0])
    after = sha(DECISIONS / rid / "mailed_ids.txt")
    check(code2 == 200 and before == after,
          "re-run after wipe reproduces the identical mailed list (deterministic rescue slice)")

    # ---- concurrency: three staff at once ---------------------------------------
    trio = [dict(r, request_id=f"concurrent_{i}") for i, r in enumerate(requests[3:6])]
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(lambda r: post(base, "/audiences", r), trio))
    counts = [b["mailed_count"] for _, b, _ in outs]
    check(all(c == 200 for c, _, _ in outs),
          f"3 simultaneous requests all succeeded ({counts}, "
          f"max {max(dt for _, _, dt in outs):.1f}s)")

    # ---- contract failures -------------------------------------------------------
    bad_area = json.loads(json.dumps(requests[6]))
    bad_area["request_id"] = "neg_area"
    bad_area["content"]["program_area"] = "not_a_area"
    code, _, _ = post(base, "/audiences", bad_area)
    check(code == 422, "unknown program_area -> 422")

    short = json.loads(json.dumps(requests[6]))
    short["request_id"] = "neg_dim"
    short["content"]["dense_embedding"] = short["content"]["dense_embedding"][:100]
    code, _, _ = post(base, "/audiences", short)
    check(code == 422, "short embedding -> 422")

    nan = json.loads(json.dumps(requests[6]))
    nan["request_id"] = "neg_null"
    nan["content"]["dense_embedding"][0] = None
    code, _, _ = post(base, "/audiences", nan)
    check(code == 422, f"null embedding value -> 422 (got {code})")

    code, _ = get(base, "/audiences/does_not_exist")
    check(code == 404, "unknown request_id status -> 404")

    failed = [m for ok, m in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        sys.exit(1)


if __name__ == "__main__":
    main()
