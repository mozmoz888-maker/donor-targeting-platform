"""Validate the 'cadence box' idea: does a person convert AT their personal rhythm?

For each repeat donor (2+ gifts) we compute their personal gap g_p = median number of
emails received between consecutive gifts. For every send after their first gift we compute
alignment = (emails since last gift) - g_p. If personal rhythms are real, conversion should
peak near alignment 0 (or -1/+1), and be materially lower far from the slot.

Also computes the aggregate hazard curve P(gift | k emails since last gift) for anyone with
a prior gift, k = 0..25+, to see whether there is any global peak at all.

Writes reports/box_validation.json.

    uv run python experiments/box_validation.py
"""

from __future__ import annotations

import bisect
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from donor_targeting.config import REPORTS_DIR, WAREHOUSE_DIR

DAY_NS = 86400 * 10**9


def main() -> None:
    tl = pq.read_table(
        WAREHOUSE_DIR / "timeline.parquet",
        columns=["type", "person_id", "content_id", "occurred_at"],
    ).to_pandas()
    tx = pq.read_table(WAREHOUSE_DIR / "transactions.parquet").to_pandas()
    tx["ts"] = tx["transaction_date"].astype("int64")
    tx = tx.sort_values(["person_id", "ts"])

    sent = tl.loc[tl["type"] == "sent", ["person_id", "content_id", "occurred_at"]].copy()
    sent["ts"] = sent["occurred_at"].astype("int64")
    sent_streams = {p: g["ts"].to_numpy() for p, g in sent.groupby("person_id")}
    tx_times = {p: g["ts"].to_numpy() for p, g in tx.groupby("person_id")}

    # donation label per (person, content) send: attributed gift within 7 days
    attributed = tx.dropna(subset=["content_id"]).astype({"content_id": "int64"})
    give_pairs = set(
        zip(
            attributed.loc[
                attributed["ts"].notna(), "person_id"
            ].to_numpy(),
            attributed["content_id"].to_numpy(),
        )
    )
    tx_by_pair: dict[tuple[int, int], list[int]] = {}
    for pid, cid, t in zip(
        attributed["person_id"].to_numpy(), attributed["content_id"].to_numpy(),
        attributed["ts"].to_numpy(),
    ):
        tx_by_pair.setdefault((pid, cid), []).append(t)

    # ---- personal gaps for repeat donors ---------------------------------------
    gaps_by_person: dict[int, float] = {}
    for p, tarr in tx_times.items():
        if len(tarr) < 2:
            continue
        sarr = sent_streams.get(p)
        if sarr is None:
            continue
        idx = [bisect.bisect_right(sarr, t) for t in tarr]
        gaps = [b - a for a, b in zip(idx[:-1], idx[1:]) if b > a]
        if gaps:
            gaps_by_person[p] = float(np.median(gaps))

    gap_stats = pd.Series(list(gaps_by_person.values()))
    print(f"repeat donors with a gap: {len(gaps_by_person):,}, "
          f"gap quartiles {gap_stats.quantile([.25, .5, .75]).round(1).tolist()}")

    # ---- per-send alignment + outcome ------------------------------------------
    rows = {"bucket": [], "donated": [], "k_since": []}
    for i, (pid, cid, ts) in enumerate(
        zip(sent["person_id"].to_numpy(), sent["content_id"].to_numpy(), sent["ts"].to_numpy())
    ):
        tarr = tx_times.get(pid)
        k_prior = bisect.bisect_right(tarr, ts) if tarr is not None else 0
        if k_prior == 0:
            continue  # no giving history yet: boxes undefined
        sarr = sent_streams[pid]
        since = bisect.bisect_left(sarr, ts) - bisect.bisect_right(sarr, tarr[k_prior - 1])
        donated = 0
        for t in tx_by_pair.get((pid, cid), ()):
            if t >= ts and t < ts + 7 * DAY_NS:
                donated = 1
                break
        rows["k_since"].append(min(since, 26))
        rows["donated"].append(donated)
        g = gaps_by_person.get(pid)
        if g is None:
            rows["bucket"].append(None)
            continue
        align = since - g
        if align <= -3:
            b = "<= -3 (early)"
        elif align == -2:
            b = "-2"
        elif align == -1:
            b = "-1"
        elif align == 0:
            b = "0 (on slot)"
        elif align == 1:
            b = "+1"
        elif align == 2:
            b = "+2"
        elif align == 3:
            b = "+3"
        else:
            b = ">= +4 (late)"
        rows["bucket"].append(b)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"sends with giving history: {n:,}, conversion {df['donated'].mean():.3%}")

    haz = df.groupby("k_since")["donated"].agg(["size", "mean"]).reset_index()
    hazard = [
        {"k_since_last_gift": int(k), "sends": int(s), "conversion": float(m)}
        for k, s, m in zip(haz["k_since"], haz["size"], haz["mean"])
    ]

    rep = df.dropna(subset=["bucket"])
    aligned = rep.groupby("bucket")["donated"].agg(["size", "mean"]).reset_index()
    order = ["<= -3 (early)", "-2", "-1", "0 (on slot)", "+1", "+2", "+3", ">= +4 (late)"]
    aligned = aligned.set_index("bucket").loc[order].reset_index()
    alignment_table = [
        {"alignment": b, "sends": int(s), "conversion": float(m)}
        for b, s, m in zip(aligned["bucket"], aligned["size"], aligned["mean"])
    ]
    print(aligned.to_string(index=False))

    on_slot = next(r for r in alignment_table if r["alignment"] == "0 (on slot)")
    late = next(r for r in alignment_table if r["alignment"] == ">= +4 (late)")
    REPORTS_DIR.mkdir(exist_ok=True)
    out = {
        "repeat_donors": int(len(gaps_by_person)),
        "personal_gap_quartiles": gap_stats.quantile([0.25, 0.5, 0.75]).round(1).tolist(),
        "sends_with_history": int(n),
        "overall_conversion_with_history": float(df["donated"].mean()),
        "hazard_by_k": hazard,
        "alignment_table": alignment_table,
        "on_slot_vs_late_lift": round(on_slot["conversion"] / late["conversion"], 3),
    }
    (REPORTS_DIR / "box_validation.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {REPORTS_DIR / 'box_validation.json'}; "
          f"on-slot vs late lift = {out['on_slot_vs_late_lift']}x")


if __name__ == "__main__":
    main()
