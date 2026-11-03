"""3D embedding evidence: PCA the 1024-dim embeddings to 3 axes and plot.

* 52 historical emails (content.dense_embedding), coloured by program area
* the 12 upcoming chatbot requests (data/requests/*.json), marked with X
* a sample of person-taste centroids (centroid_1m from the 60k feature sample),
  split into "led to a donation" vs not - donors should sit near the content they gave to

PCA is fitted on the historical email embeddings; everything else is projected into that
space. Writes a self-contained site/embeddings_3d.html (plotly.js inlined).

    uv run python experiments/embeddings_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from donor_targeting.config import REQUESTS_DIR, WAREHOUSE_DIR

SITE_DIR = Path(__file__).resolve().parent.parent / "site"
CENTROID_SAMPLE = 4000


def load_request_embeddings() -> pd.DataFrame:
    rows = []
    for path in sorted(REQUESTS_DIR.glob("*.json")):
        req = json.loads(path.read_text())
        c = req["content"]
        rows.append(
            {
                "subject": f"[new] {c['subject']}",
                "program_area": c["program_area"],
                "embedding": np.asarray(c["dense_embedding"], dtype=np.float32),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    content = pd.read_parquet(WAREHOUSE_DIR / "content.parquet")
    emb = np.stack(content["dense_embedding"].to_numpy()).astype(np.float32)

    pca = PCA(n_components=3, random_state=0)
    xyz = pca.fit_transform(emb)
    evr = pca.explained_variance_ratio_

    req = load_request_embeddings()
    req_xyz = pca.transform(np.stack(req["embedding"].to_numpy()))

    feats = pd.read_parquet(WAREHOUSE_DIR / "features.parquet")
    frame = pd.read_parquet(WAREHOUSE_DIR / "send_frame_sample.parquet")
    centroids = feats.merge(
        frame[["timeline_id", "donated"]], on="timeline_id"
    ).dropna(subset=["centroid_1m"])
    centroids = centroids.sample(n=min(CENTROID_SAMPLE, len(centroids)), random_state=0)
    cent_xyz = pca.transform(np.stack(centroids["centroid_1m"].to_numpy()))
    donated = centroids["donated"].to_numpy().astype(bool)

    fig = go.Figure()
    areas = content["program_area"].unique()
    cmap = dict(zip(areas, [f"hsl({h},70%,50%)" for h in np.linspace(0, 330, len(areas))]))

    fig.add_trace(go.Scatter3d(
        name="person taste centroids (no gift)",
        x=cent_xyz[~donated, 0], y=cent_xyz[~donated, 1], z=cent_xyz[~donated, 2],
        mode="markers",
        marker=dict(size=2, color="rgb(180,180,180)", opacity=0.35),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        name="person taste centroids (donated)",
        x=cent_xyz[donated, 0], y=cent_xyz[donated, 1], z=cent_xyz[donated, 2],
        mode="markers",
        marker=dict(size=2.5, color="rgb(255,160,0)", opacity=0.8),
        hovertemplate="donated on this send<extra></extra>",
    ))

    for area in areas:
        m = content["program_area"] == area
        fig.add_trace(go.Scatter3d(
            name=f"email: {area}",
            x=xyz[m.to_numpy(), 0], y=xyz[m.to_numpy(), 1], z=xyz[m.to_numpy(), 2],
            mode="markers",
            marker=dict(size=6, color=cmap[area], symbol="diamond"),
            text=content.loc[m, "subject"],
            hovertemplate="%{text}<extra>" + area + "</extra>",
        ))

    fig.add_trace(go.Scatter3d(
        name="NEW: upcoming requests",
        x=req_xyz[:, 0], y=req_xyz[:, 1], z=req_xyz[:, 2],
        mode="markers",
        marker=dict(size=5, color="black", symbol="x", line=dict(width=1)),
        text=req["subject"],
        hovertemplate="%{text}<extra>upcoming request</extra>",
    ))

    fig.update_layout(
        title=(
            "Donor platform embeddings in 3D (PCA of 1024-dim, "
            f"top-3 components explain {evr.sum():.0%} of variance)"
        ),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(itemsizing="constant"),
        height=820,
    )

    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "embeddings_3d.html"
    fig.write_html(out, include_plotlyjs="inline", full_html=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"PCA explained variance: {evr.round(3).tolist()}")


if __name__ == "__main__":
    main()
