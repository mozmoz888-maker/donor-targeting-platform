"""Giving cadence and the non-giver ladder.

Answers, from the full logs (verifying the history is actually there):

* data check   - is per-person giving history present and complete in the warehouse?
* cadence      - how often do donors give, in "emails received per gift" terms?
* ladder       - for each send: how many emails had the person received since their last
                 gift (or never)? conversion rate, churn rate and similarity per rung.
* suppression  - if we stop emailing people after k non-giving emails, how many donations
                 do we lose (including late first gifts) vs unsubscribes we avoid?
* ltv          - expected 6-month dollar value of an average subscriber (the EV dial).

Writes reports/cadence.json.

    uv run python experiments/cadence_analysis.py
"""

from __future__ import annotations

import bisect
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from donor_targeting.config import REPORTS_DIR, WAREHOUSE_DIR

DAY_NS = 86400 * 10**9
BIG = 10**9  # "never gave before this send"


def main() -> None:
    tl = pq.read_table(
        WAREHOUSE_DIR / "timeline.parquet",
        columns=["type", "person_id", "content_id", "occurred_at"],
    ).to_pandas()
    tl["occurred_at"] = pd.to_datetime(tl["occurred_at"], utc=True)
    tx = pq.read_table(WAREHOUSE_DIR / "transactions.parquet").to_pandas()
    people = pq.read_table(WAREHOUSE_DIR / "people.parquet").to_pandas()

    tx["ts"] = tx["transaction_date"].astype("int64")
    tx = tx.sort_values(["person_id", "ts"])
    sent = tl.loc[tl["type"] == "sent", ["person_id", "occurred_at"]].copy()
    sent["ts"] = sent["occurred_at"].astype("int64")

    sent_streams = {p: g["ts"].to_numpy() for p, g in sent.groupby("person_id")}
    tx_times: dict[int, np.ndarray] = {}
    tx_cum: dict[int, np.ndarray] = {}
    for p, g in tx.groupby("person_id"):
        tx_times[p] = g["ts"].to_numpy()
        tx_cum[p] = np.r_[0.0, g["amount"].to_numpy().cumsum()]

    # ---- data check: is the giving history actually in the logs? ----------------
    people_ids = set(people["id"].tolist())
    attributed = tx.dropna(subset=["content_id"])
    attributed_people = set(attributed["person_id"].tolist())
    givers = tx.groupby("person_id").size()
    data_check = {
        "transactions": int(len(tx)),
        "total_donated_usd": round(float(tx["amount"].sum()), 2),
        "share_tx_persons_known": float(np.mean(list(map(people_ids.__contains__, givers.index)))),
        "share_attributed_to_email": float(len(attributed) / len(tx)),
        "givers": int(len(givers)),
        "share_givers_with_2plus_gifts": float((givers >= 2).mean()),
        "share_givers_with_an_attributed_gift": float(
            np.mean(list(map(attributed_people.__contains__, givers.index)))
        ),
        "people_total": int(len(people)),
        "people_with_sends": int(len(sent_streams)),
        "max_gifts_per_person": int(givers.max()),
    }
    print("data check:", data_check)

    # ---- person-level cadence ---------------------------------------------------
    rows = []
    for p, sarr in sent_streams.items():
        tarr = tx_times.get(p)
        rec = {"person_id": p, "n_sends": len(sarr), "n_gifts": 0 if tarr is None else len(tarr)}
        if tarr is not None and len(tarr):
            rec["sends_since_gift"] = len(sarr) - bisect.bisect_right(sarr, tarr[-1])
            rec["sends_before_first_gift"] = bisect.bisect_left(sarr, tarr[0])
            rec["total_amount"] = float(tx_cum[p][-1])
        rows.append(rec)
    per = pd.DataFrame(rows)
    givers_frame = per[per["n_gifts"] > 0]

    ratio = (givers_frame["n_gifts"] / givers_frame["n_sends"]).replace([np.inf], np.nan)
    gaps = []
    for p in givers_frame["person_id"]:
        sarr, tarr = sent_streams[p], tx_times[p]
        if len(tarr) >= 2:
            idx = [bisect.bisect_right(sarr, t) for t in tarr]
            gaps += [b - a for a, b in zip(idx[:-1], idx[1:])]
    first_idx = givers_frame["sends_before_first_gift"]

    ever_gave = float((per["n_gifts"] > 0).mean())
    mean_gifts_per_giver = float(givers_frame["n_gifts"].mean())
    mean_amount = float(tx["amount"].mean())
    ltv = {
        "ever_gave_share": ever_gave,
        "mean_gifts_per_giver_6mo": mean_gifts_per_giver,
        "mean_gift_usd": mean_amount,
        "median_gift_usd": float(tx["amount"].median()),
        "expected_6mo_value_per_subscriber_usd": round(
            ever_gave * mean_gifts_per_giver * mean_amount, 2
        ),
    }
    cadence = {
        "givers": int(len(givers_frame)),
        "gifts_per_send_among_givers": {
            "median": float(ratio.median()), "q25": float(ratio.quantile(0.25)),
            "q75": float(ratio.quantile(0.75)),
        },
        "median_emails_between_gifts": float(np.median(gaps)) if gaps else None,
        "emails_between_gifts_quartiles": (
            [float(np.quantile(gaps, q)) for q in (0.25, 0.5, 0.75)] if gaps else None
        ),
        "first_gift_after_n_non_giving_sends": {
            "median": float(first_idx.median()),
            "share_at_5plus": float((first_idx >= 5).mean()),
            "share_at_10plus": float((first_idx >= 10).mean()),
            "share_at_20plus": float((first_idx >= 20).mean()),
        },
    }
    print("cadence:", cadence)
    print("ltv:", ltv)

    # ---- send-level ladder on the 60k sample (has similarity + outcomes) --------
    frame = pd.read_parquet(WAREHOUSE_DIR / "send_frame_sample.parquet")
    since, prior_gifts = [], []
    for person, ts in zip(
        frame["person_id"].to_numpy(), frame["sent_at"].astype("int64").to_numpy()
    ):
        sarr = sent_streams[person]
        prior_sends = bisect.bisect_left(sarr, ts)
        tarr = tx_times.get(person)
        if tarr is None:
            since.append(BIG)
            prior_gifts.append(0)
            continue
        k = bisect.bisect_right(tarr, ts)  # gifts strictly before this send
        prior_gifts.append(k)
        if k == 0:
            since.append(BIG)
        else:
            since.append(prior_sends - bisect.bisect_right(sarr, tarr[k - 1]))
    frame["emails_since_gift"] = since
    frame["prior_gifts"] = prior_gifts

    def rung(v: int) -> str:
        if v >= BIG:
            return "never gave"
        if v <= 1:
            return "0-1"
        if v <= 3:
            return "2-3"
        if v <= 5:
            return "4-5"
        if v <= 10:
            return "6-10"
        return "10+"

    frame["rung"] = frame["emails_since_gift"].map(rung)
    g = frame.groupby("rung", observed=True)
    ladder = pd.DataFrame({
        "sends": g.size(),
        "donation_rate": g["donated"].mean(),
        "unsub_rate": g["unsubscribed"].mean(),
        "mean_similarity": g["centroid_cosine_similarity"].mean(),
        "mean_eng_events_30d": g["eng_events_30d"].mean(),
    }).reset_index()
    order = ["never gave", "0-1", "2-3", "4-5", "6-10", "10+"]
    ladder = ladder.set_index("rung").loc[order].reset_index()
    print(ladder.to_string(index=False))

    # ---- suppression simulation: stop emailing after k non-giving emails --------
    never = frame["emails_since_gift"] >= BIG
    suppression = []
    for k in (3, 5, 10):
        suppress = (frame["emails_since_gift"] > k) | never
        suppress = suppress.to_numpy()
        suppression.append({
            "k": k,
            "sends_suppressed": float(suppress.mean()),
            "donations_lost": float(
                (suppress & frame["donated"].to_numpy()).sum() / max(frame["donated"].sum(), 1)
            ),
            "first_gifts_lost": float(
                (suppress & frame["donated"].to_numpy() & (frame["prior_gifts"] == 0).to_numpy()).sum()
                / max(((frame["prior_gifts"] == 0) & frame["donated"]).sum(), 1)
            ),
            "unsubs_avoided": float(
                (suppress & frame["unsubscribed"].to_numpy()).sum()
                / max(frame["unsubscribed"].sum(), 1)
            ),
        })
    for s in suppression:
        print(s)

    REPORTS_DIR.mkdir(exist_ok=True)
    out = {
        "data_check": data_check,
        "cadence": cadence,
        "ltv": ltv,
        "ladder": ladder.to_dict("records"),
        "suppression": suppression,
    }
    (REPORTS_DIR / "cadence.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {REPORTS_DIR / 'cadence.json'}")


if __name__ == "__main__":
    main()
