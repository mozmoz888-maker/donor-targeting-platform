"""Churn investigation: why do 1-2% of recipients unsubscribe from each email?

Builds a send-level frame over the warehouse (pandas + per-person event streams, no SQL) and
measures how unsubscribe risk varies by:

* program area of the email (full population)
* subscriber tenure (full population)
* engagement in the prior 30 days (60k random send sample, exact per-person histories)
* email frequency: sends received in the prior 30 days (sample)
* prior donation history (sample)
* centroid similarity to the person's recent taste (sample)

Writes reports/analysis.json (rendered to reports/ANALYSIS.md by render_report.py).

    uv run python experiments/churn_analysis.py
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from donor_targeting.config import REPORTS_DIR, WAREHOUSE_DIR
from donor_targeting.warehouse_io import modelling_frame

DAY_NS = 86400 * 10**9


def load_timeline(path: Path) -> pd.DataFrame:
    tl = pq.read_table(path, columns=["type", "person_id", "content_id", "occurred_at"]).to_pandas()
    tl["occurred_at"] = pd.to_datetime(tl["occurred_at"], utc=True)
    return tl


def streams(g: pd.DataFrame) -> dict[int, np.ndarray]:
    """person -> sorted int64 timestamps of their events."""
    return {
        p: grp["ts"].to_numpy()
        for p, grp in g.assign(ts=g["occurred_at"].astype("int64")).groupby("person_id")
    }


def main() -> None:
    tl = load_timeline(WAREHOUSE_DIR / "timeline.parquet")
    people = pq.read_table(WAREHOUSE_DIR / "people.parquet").to_pandas()
    content = pq.read_table(WAREHOUSE_DIR / "content.parquet",
                            columns=["id", "program_area"]).to_pandas()
    tx = pq.read_table(WAREHOUSE_DIR / "transactions.parquet").to_pandas()
    tx["ts"] = tx["transaction_date"].astype("int64")
    tx = tx.sort_values(["person_id", "ts"])

    # ---- full-population frame: sends + unsubscribe label + context -------------
    sents = tl.loc[tl["type"] == "sent", ["person_id", "content_id", "occurred_at"]].copy()
    unsub_pairs = tl.loc[tl["type"] == "unsubscribed", ["person_id", "content_id"]].drop_duplicates()
    opened_pairs = tl.loc[tl["type"] == "opened", ["person_id", "content_id"]].drop_duplicates()
    sents = sents.merge(unsub_pairs.assign(unsubscribed=True),
                        on=["person_id", "content_id"], how="left")
    sents["unsubscribed"] = sents["unsubscribed"].fillna(False)
    sents = sents.merge(opened_pairs.assign(opened_this_email=True),
                        on=["person_id", "content_id"], how="left")
    sents["opened_this_email"] = sents["opened_this_email"].fillna(False)

    program = dict(zip(content["id"], content["program_area"]))
    sents["program_area"] = sents["content_id"].map(program)
    sub_at = pd.Series(people["subscribed_at"].astype("int64").to_numpy(),
                       index=people["id"].to_numpy())
    sents["tenure_days"] = (
        (sents["occurred_at"].astype("int64") - sents["person_id"].map(sub_at)) / DAY_NS
    )

    overall = {
        "sends": int(len(sents)),
        "people": int(sents["person_id"].nunique()),
        "emails": int(sents["content_id"].nunique()),
        "unsubscribed": int(sents["unsubscribed"].sum()),
        "unsub_rate": float(sents["unsubscribed"].mean()),
    }
    print("overall:", overall)

    def rate_table(df: pd.DataFrame, by: str) -> list[dict]:
        g = df.groupby(by, dropna=False, observed=True)
        out = pd.DataFrame(
            {"sends": g.size(), "unsub_rate": g["unsubscribed"].mean()}
        ).reset_index()
        return out.sort_values("unsub_rate", ascending=False).to_dict("records")

    by_program = rate_table(sents, "program_area")
    tenure = sents.assign(tenure=pd.cut(
        sents["tenure_days"], [-1, 30, 90, 180, 10**9],
        labels=["<30d", "30-90d", "90-180d", "180d+"],
    ))
    by_tenure = rate_table(tenure, "tenure")
    opened_of_unsub = float(sents.loc[sents["unsubscribed"], "opened_this_email"].mean())

    # ---- 60k random send sample with outcomes (repo logic) + exact histories ----
    frame = modelling_frame()  # timeline_id, centroid_cosine_similarity, person_id,
    #                          # content_id, sent_at, amount, donated, unsubscribed
    eng_streams = streams(tl.loc[tl["type"].isin(["opened", "clicked"])])
    sent_streams = streams(tl.loc[tl["type"] == "sent"])
    tx_times: dict[int, np.ndarray] = {}
    tx_cum: dict[int, np.ndarray] = {}
    for p, grp in tx.groupby("person_id"):
        tx_times[p] = grp["ts"].to_numpy()
        tx_cum[p] = np.r_[0.0, grp["amount"].to_numpy().cumsum()]

    sent_ts = frame["sent_at"].astype("int64").to_numpy()
    eng_30d, days_eng, sends_30d, prior_n, prior_amt = [], [], [], [], []
    for person, ts in zip(frame["person_id"].to_numpy(), sent_ts):
        arr = eng_streams.get(person, ())
        i, j = bisect.bisect_left(arr, ts), bisect.bisect_left(arr, ts - 30 * DAY_NS)
        eng_30d.append(i - j)
        days_eng.append((ts - arr[i - 1]) / DAY_NS if i > 0 else np.nan)
        sarr = sent_streams.get(person, ())
        si, sj = bisect.bisect_left(sarr, ts), bisect.bisect_left(sarr, ts - 30 * DAY_NS)
        sends_30d.append(si - sj)
        tarr = tx_times.get(person, ())
        k = bisect.bisect_left(tarr, ts)
        prior_n.append(k)
        prior_amt.append(float(tx_cum.get(person, (0.0,))[k]) if k < len(tx_cum.get(person, ())) else 0.0)

    frame["eng_events_30d"] = eng_30d
    frame["days_since_eng"] = days_eng
    frame["sends_30d"] = sends_30d
    frame["prior_donations"] = prior_n
    frame["prior_amount"] = prior_amt
    frame["prior_donor"] = frame["prior_donations"] > 0
    frame["has_eng_30d"] = frame["eng_events_30d"] > 0
    frame["tenure_days"] = (sent_ts - frame["person_id"].map(sub_at).to_numpy()) / DAY_NS

    overall_sample = {
        "sample_sends": int(len(frame)),
        "unsub_rate": float(frame["unsubscribed"].mean()),
        "donation_rate": float(frame["donated"].mean()),
        "median_gift": float(frame.loc[frame["amount"] > 0, "amount"].median()),
    }

    eng = frame.assign(engagement=pd.cut(
        frame["eng_events_30d"], [-1, 0, 2, 5, 10**9],
        labels=["none", "1-2", "3-5", "6+"],
    ))
    by_engagement = rate_table(eng, "engagement")

    freq = frame.assign(freq=pd.cut(
        frame["sends_30d"], [-1, 5, 8, 10, 12, 10**9],
        labels=["<=5", "6-8", "9-10", "11-12", "13+"],
    ))
    by_frequency = rate_table(freq, "freq")
    by_donor = rate_table(frame, "prior_donor")

    sample = frame.dropna(subset=["centroid_cosine_similarity"]).copy()
    sample["sim_decile"] = pd.qcut(sample["centroid_cosine_similarity"], 10, labels=False) + 1
    by_sim = rate_table(sample.assign(donated=sample["donated"].astype(bool)), "sim_decile")
    # add donation rate per decile for the same table
    g = sample.groupby("sim_decile")["donated"].mean()
    for row in by_sim:
        row["donation_rate"] = float(g.loc[row["sim_decile"]])

    # policy simulation on the sample (same methodology as baseline.py)
    order = sample.sort_values("centroid_cosine_similarity", ascending=False).reset_index(
        drop=True
    )
    n = len(order)
    pol = [{"policy": "BAU: random 75%", "emailed": 0.75,
            "unsub_caused": 0.75, "donations_kept": 0.75, "dollars_kept": 0.75}]
    for frac in (0.75, 0.50, 0.25):
        k = int(frac * n)
        m = np.zeros(n, dtype=bool)
        m[:k] = True
        pol.append({
            "policy": f"centroid top {frac:.0%}",
            "emailed": float(m.mean()),
            "unsub_caused": float(order["unsubscribed"].to_numpy()[m].sum()
                                  / max(order["unsubscribed"].sum(), 1)),
            "donations_kept": float(order["donated"].to_numpy()[m].sum()
                                    / max(order["donated"].sum(), 1)),
            "dollars_kept": float(order["amount"].to_numpy()[m].sum()
                                  / max(order["amount"].sum(), 1e-9)),
        })

    frame.to_parquet(WAREHOUSE_DIR / "send_frame_sample.parquet", index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    result = {
        "overall": overall,
        "overall_sample": overall_sample,
        "opened_this_email_when_unsubscribed": opened_of_unsub,
        "by_program_area": by_program,
        "by_tenure": by_tenure,
        "by_engagement_30d": by_engagement,
        "by_frequency_30d": by_frequency,
        "by_prior_donor": by_donor,
        "by_similarity_decile": by_sim,
        "policy_simulation": pol,
    }
    (REPORTS_DIR / "analysis.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {REPORTS_DIR / 'analysis.json'} and send_frame_sample.parquet")


if __name__ == "__main__":
    main()
