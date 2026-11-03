"""Nightly precompute: rebuild per-person scoring state after the warehouse refresh.

Runs in the 01:00-09:00 ET window (cron / Render job). For every ACTIVE subscriber:

* centroid_30d - mean embedding of the distinct content they engaged with in the 30 days
  before the warehouse snapshot (float16, memory-mapped at request time)
* centroid_norm - its L2 norm (cosines stay exact)
* emails_since_gift - sends received since their last gift (-1 = never gave)

Chunked so peak RSS stays well under the 2 GB worker limit. Writes data/artifacts/.

    uv run python -m service.precompute
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from donor_targeting.config import CENTROID_LOOKBACK_DAYS, SNAPSHOT_AT, WAREHOUSE_DIR
from service import config


def build(warehouse_dir=config.ARTIFACTS_DIR) -> dict:
    t0 = time.time()
    people = pq.read_table(WAREHOUSE_DIR / "people.parquet").to_pandas()
    content = pq.read_table(WAREHOUSE_DIR / "content.parquet").to_pandas()
    tl = pq.read_table(
        WAREHOUSE_DIR / "timeline.parquet",
        columns=["type", "person_id", "content_id", "occurred_at"],
    ).to_pandas()
    tl["ts"] = tl["occurred_at"].astype("int64")

    active = people.loc[people["status"] == "active", ["id"]].rename(columns={"id": "person_id"})
    active = active.sort_values("person_id").reset_index(drop=True)
    print(f"{len(active):,} active subscribers")

    emb = np.stack(content["dense_embedding"].to_numpy()).astype(np.float32)
    content_ids = content["id"].to_numpy()
    col_of = {c: i for i, c in enumerate(content_ids)}

    # engagement in the 30 days before the snapshot, deduped to distinct content
    eng = tl.loc[
        (tl["type"].isin(["opened", "clicked"]))
        & (tl["ts"] >= int(pd.Timestamp(SNAPSHOT_AT).value) - CENTROID_LOOKBACK_DAYS * 86400 * 10**9)
        & (tl["ts"] < int(pd.Timestamp(SNAPSHOT_AT).value)),
        ["person_id", "content_id"],
    ].drop_duplicates()
    eng = eng[eng["person_id"].isin(set(active["person_id"]))]
    eng["col"] = eng["content_id"].map(col_of).astype("int64")
    row_of = pd.Series(np.arange(len(active)), index=active["person_id"])
    eng["row"] = eng["person_id"].map(row_of).astype("int64")
    counts = np.bincount(eng["row"].to_numpy(), minlength=len(active))
    weights = 1.0 / np.maximum(counts[eng["row"].to_numpy()], 1)
    w = sparse.csr_matrix(
        (weights, (eng["row"].to_numpy(), eng["col"].to_numpy())),
        shape=(len(active), len(content_ids)),
    )
    has_centroid = counts > 0
    print(f"{has_centroid.sum():,} with a 30-day taste centroid "
          f"({1 - has_centroid.mean():.0%} dormant)")

    # chunked centroids -> float16 on disk (never materialise the full float32 matrix)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cent_path = config.ARTIFACTS_DIR / "centroids.f16.npy"
    n_rows, n_cols = w.shape
    centroids = np.lib.format.open_memmap(
        cent_path, mode="w+", dtype=np.float16, shape=(n_rows, config.EMBEDDING_DIM)
    )
    norms = np.zeros(n_rows, dtype=np.float32)
    for s in range(0, n_rows, config.PRECOMPUTE_CHUNK):
        e = min(s + config.PRECOMPUTE_CHUNK, n_rows)
        block = np.asarray(w[s:e] @ emb, dtype=np.float32)
        norms[s:e] = np.linalg.norm(block, axis=1)
        centroids[s:e] = block.astype(np.float16)
    centroids.flush()

    # emails since last gift (-1 = never gave)
    tx = pq.read_table(WAREHOUSE_DIR / "transactions.parquet",
                       columns=["person_id", "transaction_date"]).to_pandas()
    tx["ts"] = tx["transaction_date"].astype("int64")
    last_gift = tx.groupby("person_id")["ts"].max()
    sent = tl.loc[tl["type"] == "sent", ["person_id", "ts"]]
    sent_counts = sent.groupby("person_id").size()
    snap = int(pd.Timestamp(SNAPSHOT_AT).value)

    # sends after the last gift per person
    sent_sorted = sent.sort_values(["person_id", "ts"])
    sent_by_person = {p: g["ts"].to_numpy() for p, g in sent_sorted.groupby("person_id")}
    since = np.full(len(active), -1, dtype=np.int32)
    for i, p in enumerate(active["person_id"].to_numpy()):
        lg = last_gift.get(p)
        if pd.notna(lg):
            arr = sent_by_person.get(p, np.array([], dtype="int64"))
            since[i] = len(arr) - int(np.searchsorted(arr, int(lg), side="right"))

    persons = pd.DataFrame(
        {"person_id": active["person_id"].to_numpy(), "has_centroid": has_centroid,
         "emails_since_gift": since}
    )
    persons.to_parquet(config.ARTIFACTS_DIR / "persons.parquet", index=False)
    np.save(config.ARTIFACTS_DIR / "norms.f32.npy", norms)

    meta = {
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "active_people": int(len(active)),
        "with_centroid": int(has_centroid.sum()),
        "embedding_dim": config.EMBEDDING_DIM,
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "build_seconds": round(time.time() - t0, 1),
        "model_sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest(),
    }
    (config.ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"artifacts written to {config.ARTIFACTS_DIR} in {time.time() - t0:.0f}s: {meta}")
    return meta


if __name__ == "__main__":
    build()
