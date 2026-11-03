# DESIGN.md — donor targeting service

**What this is.** A production API that takes the routing agent's request (an email's
1024-dim embedding + program area) and, within 3 minutes, has in the CRM the list of people
who should receive it. It ships the data science team's logistic model under the business
rule we validated against six months of history. All numbers below are measured, not
estimated; rerun everything with `make analyze` and `make evidence`.

## Architecture: what happens ahead of time vs at request time

```text
nightly, 01:00-09:00 ET window          request time (SLA 180 s)
─────────────────────────────           ─────────────────────────────────────
warehouse refresh (given)               POST /audiences
  └─ make precompute (cron/Render job)    ├─ validate (422 fast-fail on bad input)
       ├─ centroids.f16.npy  (244k x      ├─ score 244k actives vs embedding
       │    1024, memmapped, 0.5 GB)      │    (cosine -> model p_donate, 0.4 s)
       ├─ persons.parquet (ids, dormancy, ├─ EV rule -> mail / suppress + reasons
       │    emails-since-gift)            ├─ push list to CRM (1 request, ~2 s,
       ├─ norms + meta.json (model        │    retry 10/20/40/80 s under deadline)
       │    sha256 pinned)                ├─ decision log + mailed_ids.txt
       └─ 9 s end to end                  └─ respond {list_id, counts, timings}
```

**Latency budget for a 244,057-person request (measured, `reports/SERVICE.md`):**
validation ~1 ms, scoring **0.38 s**, model inference included, CRM push **2.0 s**,
logging ~10 ms → **~2.5 s end to end, 1.4% of the 180 s SLA**. The headroom is the point:
the budget survives a 100x slower warehouse, model, or CRM.

**Memory budget (measured):** the worker holds the person table (~3 MB), norms (1 MB) and
a memory-mapped float16 centroid matrix (0.5 GB on disk, paged in 50k-row chunks during
scoring). Peak RSS across all evidence scenarios incl. chaos runs: **1,145 MB of the
2,048 MB limit** (`reports/service_evidence.json`) — most of it the centroid pages read
during the first score (subsequent requests reuse them) plus interpreter/sklearn. The full
float32 matrix would be 1.0 GB on top of that; float16 + mmap is what keeps the worker
inside 2 GB with margin. Cosine on fp16 introduces error ~1e-3, three orders below the
feature's decision range.

## The audience rule, and the trade-off it makes

Per active person *p* and requested email *e*: **mail iff**
`p_donate(p,e) × $57.94 ≥ p_unsub × LTV + $0.002`
where `p_donate` is the DS team's logistic model on `centroid_cosine_similarity`
(out-of-time AUC 0.63; NULL → imputer median + was-null indicator), `p_unsub = 1.5%`
(churn proved barely predictable — AUC 0.57 best of four model families — so we price it
as a constant), and **LTV = $19.62** = the measured expected 6-month value of a subscriber
(25.8% ever give × 1.31 gifts × $57.94; `reports/cadence.json`).

Why this shape, from the evidence in `reports/ANALYSIS.md`:

- Similarity is the one lever that moves both sides: decile-10 sends churn 1.09% vs 1.7-2.0%
  in the bottom deciles and convert 3.5% vs ~0.5%. At BAU-average conversion (1.15%) a send
  earns $0.67 against $0.29 of churn cost; at the bottom decile it is break-even. The EV
  rule simply refuses to send break-even emails.
- **We do not suppress by giving history.** Conversion does not decay with non-giving streak
  length (it rises: 1.17% at 0-1 since gift → 1.72% at 6-10), first gifts arrive after a
  median 13 silent sends, and "stop after k non-giving emails" kills 94-100% of first gifts.
  Personal "rhythm slots" were tested and rejected: no conversion peak at a person's slot
  (`reports/box_validation.json`).
- **Dormancy rescue.** 25% of actives have no 30-day engagement (unknown taste). They never
  enter via EV; instead each request deterministically includes 15% of them
  (hash of request_id+person_id, so replays are stable and idempotency holds). This keeps a
  re-engagement path open so the policy cannot permanently starve future donors.

**Business outcome at defaults** (191,441 mailed for a typical request, inside the 100k-250k
band): by the baseline replay, ranking by similarity and mailing ~75% of the list keeps
~88% of donations while causing ~69% of BAU's unsubscribes — i.e. the same gifts for about
8% less churn. The LTV dial is steep (measured, same request): $19.62 → 191,402 mailed
(0.51% cutoff); $40 → 9,654 mailed (1.04% cutoff) — the model's probabilities pack tightly
between 0.5% and 1.5%, so audience size swings hard once LTV crosses ~$25. That is the
business decision we expose, not one we make.

## Persistence

| data | where | why |
| --- | --- | --- |
| scoring artifacts | `data/artifacts/` (fp16 centroids, norms, persons, meta) | rebuilt nightly; memmapped at request time; meta pins model sha256 |
| decision record | `data/decisions/<request_id>/decision.json` + `index.jsonl` | audit trail, status endpoint, idempotency, future labels |
| mailed ids | `data/decisions/<request_id>/mailed_ids.txt` | survives CRM failure for later re-push; replay evidence |
| CRM outbox | `data/crm_outbox/` (mock CRM) | the lists themselves, scored from there |

Nothing is persisted at request time except the decision; the warehouse is never read
after precompute.

## What can go wrong, and what we do

| failure | behaviour |
| --- | --- |
| malformed request (schema, dim, unknown program area, non-finite embedding) | 422 immediately; retries cannot fix bad input |
| duplicate request_id | 409 + the original result; a request can never create two CRM lists |
| CRM transient failure | retry 10/20/40/80 s, governed by the SLA clock: an attempt only starts if it can finish with 15 s margin; recovered case measured at 3 attempts / ~34 s |
| CRM down for good | graceful terminal FAILED (503 + status endpoint), audience persisted for re-push; ladder ends ~160 s < 180 s |
| warehouse refresh late/failed | serve yesterday's artifacts; `/healthz` reports artifact age and `staleness_ok` (32 h threshold) for alerting |
| model file tampered/replaced | refused at load: sha256 pinned in `meta.json` at precompute time |
| worker OOM | not reachable at current shapes: 600 MB peak vs 2 GB, chunked scoring, mmap; the container enforces the same limit |

## How we know it works live

`/healthz` (artifact freshness, active counts), the decision log (per request: audience
size, suppress reasons, timings, attempts), and after each send the CRM outbox is the
ground truth (no duplicates, all actives — asserted in the evidence suite). The metrics
that matter weekly: donations and $ per email vs BAU prediction, unsubscribes per email vs
the 1.5% base, and the dormant-share of the list (a rising dormant share is the starvation
warning). The full replay + chaos evidence regenerates with `make evidence`.

## What I would do next

1. **Backtest-on-send loop:** persist per-person p_donate with each list and join outcomes
   after 7 days — live calibration instead of frozen offline metrics.
2. **Dormancy policy as a first-class experiment:** the 15% rescue slice is randomised, so
   it doubles as the holdout that measures the counterfactual (what suppressed people
   would have done).
3. Model refresh in the nightly job (retrain on the rolling window; the artifacts/hash
   chain already supports versioning).
4. Real CRM adapter behind the same `push_with_retry` interface (the mock is in-process;
   an HTTP adapter adds URL allowlisting and request signing).
5. Queue-based intake (Redis/SQS) if request volume ever makes the 8-hour precompute window
   or concurrent scoring worth revisiting — at 2 sends/week it is not close.
