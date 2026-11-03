"""Render reports/ANALYSIS.md and site/index.html from the JSON results.

    uv run python experiments/render_report.py

Reads reports/analysis.json and reports/model_runs.json (written by churn_analysis.py and
model_runs.py) and produces:
  * reports/ANALYSIS.md  - full markdown report (also copied into site/)
  * site/index.html      - web dashboard with the same tables + link to the 3D embeddings
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
SITE = REPO / "site"


def md_table(rows: list[dict], cols: list[tuple[str, str]], fmts: dict[str, str] | None = None) -> str:
    fmts = fmts or {}
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            f = fmts.get(key)
            cells.append(f.format(v) if f else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def build_sections(analysis: dict, runs: dict) -> tuple[list[tuple[str, list[str]]], str]:
    """Returns (sections as (title, [markdown lines]), executive summary)."""
    ov, ovs = analysis["overall"], analysis["overall_sample"]
    worst = analysis["by_program_area"][0]
    best = analysis["by_program_area"][-1]
    sim = {r["sim_decile"]: r for r in analysis["by_similarity_decile"]}
    top, bottom = sim[10], sim[1]
    pol = {p["policy"]: p for p in analysis["policy_simulation"]}
    p50 = pol["centroid top 50%"]

    summary = (
        f"- **{ov['unsub_rate']:.2%} of sends cause an unsubscribe** "
        f"({ov['unsubscribed']:,} of {ov['sends']:,} sends to {ov['people']:,} people). "
        f"Donation rate is {ovs['donation_rate']:.2%}, median gift ${ovs['median_gift']:.0f}.\n"
        f"- **Content relevance is the dominant churn lever.** Sends in similarity decile 10 "
        f"(email matches the person's recent taste) unsubscribe at {bottom and top['unsub_rate']:.2%} "
        f"vs {bottom['unsub_rate']:.2%} in decile 1 — and donate {top['donation_rate']:.2%} vs "
        f"{bottom['donation_rate']:.2%} ({top['donation_rate'] / bottom['donation_rate']:.1f}x).\n"
        f"- **Program area matters**: {worst['program_area']} churns worst "
        f"({worst['unsub_rate']:.2%}/send), {best['program_area']} best ({best['unsub_rate']:.2%}).\n"
        f"- **Engagement protects**: no engagement in 30d = "
        f"{analysis['by_engagement_30d'][0]['unsub_rate']:.2%} churn vs "
        f"{analysis['by_engagement_30d'][-1]['unsub_rate']:.2%} for the most engaged.\n"
        f"- Only {analysis['opened_this_email_when_unsubscribed']:.0%} of unsubscribes even opened "
        f"the email they left from — churn is background fatigue, not a reaction to content read.\n"
        f"- **Targeting works**: emailing the top 50% by similarity causes {p50['unsub_caused']:.0%} "
        f"of unsubscribes while keeping {p50['donations_kept']:.0%} of donations "
        f"(vs 75%/75% for BAU). Among engaged sends only."
    )

    f3 = {"unsub_rate": "{:" + pct_fmt + "}", "donation_rate": "{:" + pct_fmt + "}"}
    sections = [
        ("Why subscribers leave: risk factors", [
            "**By program area** (full population)", "",
            md_table(analysis["by_program_area"],
                     [("program_area", "program area"), ("sends", "sends"),
                      ("unsub_rate", "unsub rate / send")],
                     {"unsub_rate": "{:.2%}"}), "",
            "**By prior 30-day engagement** (60k sample, exact histories)", "",
            md_table(analysis["by_engagement_30d"],
                     [("engagement", "engagement events"), ("sends", "sends"),
                      ("unsub_rate", "unsub rate")],
                     {"unsub_rate": "{:.2%}"}), "",
            "**By email frequency** (sends received in prior 30 days)", "",
            md_table(analysis["by_frequency_30d"],
                     [("freq", "sends in 30d"), ("sends", "sends in sample"),
                      ("unsub_rate", "unsub rate")],
                     {"unsub_rate": "{:.2%}"}), "",
            "**By subscriber tenure**", "",
            md_table(analysis["by_tenure"],
                     [("tenure", "tenure"), ("sends", "sends"),
                      ("unsub_rate", "unsub rate")],
                     {"unsub_rate": "{:.2%}"}), "",
            "**By prior donation history**", "",
            md_table(analysis["by_prior_donor"],
                     [("prior_donor", "donated before"), ("sends", "sends"),
                      ("unsub_rate", "unsub rate")],
                     {"unsub_rate": "{:.2%}"}), "",
        ]),
        ("Embedding similarity: churn down, donations up", [
            "Each sampled send, bucketed into deciles of `centroid_cosine_similarity` "
            "(how closely the email matches the person's recent engagement):", "",
            md_table(analysis["by_similarity_decile"],
                     [("sim_decile", "decile (10 = best match)"), ("sends", "sends"),
                      ("unsub_rate", "unsub rate"), ("donation_rate", "donation rate")],
                     {"unsub_rate": "{:.2%}", "donation_rate": "{:.2%}"}), "",
            "**Policy simulation** (same methodology as `baseline.py`, engaged sends only):", "",
            md_table(analysis["policy_simulation"],
                     [("policy", "policy"), ("emailed", "% emailed"),
                      ("unsub_caused", "% of unsubscribes caused"),
                      ("donations_kept", "% of donations kept"),
                      ("dollars_kept", "% of $ kept")],
                     {"emailed": "{:.0%}", "unsub_caused": "{:.1%}",
                      "donations_kept": "{:.1%}", "dollars_kept": "{:.1%}"}),
        ]),
    ]

    for target, label in (("donated", "Donation model (production target)"),
                          ("unsubscribed", "Churn model")):
        t = runs["targets"][target]
        rows = []
        for m in t["models"]:
            rows.append({
                "model": m["model"],
                "roc_auc": f"{m['roc_auc']:.3f}",
                "avg_precision": f"{m['average_precision']:.3f}",
                "top50": pct(m.get("top50_donations_kept",
                                   m.get("top50_unsubs_caused", 0))),
            })
        lines = [
            f"Base rate in holdout: {t['base_rate_holdout']:.2%}. "
            f"Out-of-time split: holdout from {runs['holdout_from']}.", "",
            md_table(rows, [("model", "model"), ("roc_auc", "ROC AUC"),
                            ("avg_precision", "avg precision"),
                            ("top50", "kept/caused @ top 50%")]), "",
        ]
        best_forest = next((m for m in t["models"]
                            if "feature_importances" in m
                            and "forest" in m["model"]), None)
        if best_forest:
            fi = sorted(best_forest["feature_importances"].items(),
                        key=lambda kv: -kv[1])
            lines += [
                f"Feature importances ({best_forest['model']}): "
                + ", ".join(f"`{k}` {v:.2f}" for k, v in fi), "",
            ]
        sections.append((label, lines))

    cad_path = REPORTS / "cadence.json"
    if cad_path.exists():
        cad = json.loads(cad_path.read_text())
        dc, ca, lt = cad["data_check"], cad["cadence"], cad["ltv"]
        fb = ca["first_gift_after_n_non_giving_sends"]
        q = ca["emails_between_gifts_quartiles"] or [0, 0, 0]
        sections.append(("Giving cadence: the logs have the history", [
            f"Data check: {dc['transactions']:,} transactions from {dc['givers']:,} givers "
            f"(${dc['total_donated_usd']:,.0f} total), {dc['share_givers_with_2plus_gifts']:.0%} "
            f"of givers gave 2+ times, {dc['share_givers_with_an_attributed_gift']:.0%} have at "
            f"least one email-attributed gift — **the per-person giving history is fully in the "
            f"logs**, nothing was stripped out.", "",
            f"- Repeat donors give a gift every **{ca['median_emails_between_gifts']:.0f} emails "
            f"received** (quartiles {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}). Median donor converts for "
            f"the first time after **{fb['median']:.0f} non-giving sends**; "
            f"{fb['share_at_10plus']:.0%} of all first gifts arrive after 10+ non-giving sends, "
            f"{fb['share_at_20plus']:.0%} after 20+.", "",
            f"- Subscriber value: {lt['ever_gave_share']:.1%} ever give, averaging "
            f"{lt['mean_gifts_per_giver_6mo']:.1f} gifts of ${lt['mean_gift_usd']:.0f} (mean) in "
            f"six months — an expected **${lt['expected_6mo_value_per_subscriber_usd']:.2f} per "
            f"subscriber per 6 months**. Losing one costs the mission that, plus future years.", "",
            "**The non-giver ladder** (per send: how many emails since the person last gave):", "",
            md_table(cad["ladder"],
                     [("rung", "emails since last gift"), ("sends", "sends in sample"),
                      ("donation_rate", "donation rate"), ("unsub_rate", "unsub rate"),
                      ("mean_similarity", "mean similarity"),
                      ("mean_eng_events_30d", "mean engagement")],
                     {"donation_rate": "{:.2%}", "unsub_rate": "{:.2%}",
                      "mean_similarity": "{:.3f}", "mean_eng_events_30d": "{:.1f}"}), "",
            "**If we stop emailing people after k non-giving emails** (sample simulation):", "",
            md_table(cad["suppression"],
                     [("k", "k"), ("sends_suppressed", "% sends suppressed"),
                      ("donations_lost", "% donations lost"),
                      ("first_gifts_lost", "% of first gifts lost"),
                      ("unsubs_avoided", "% unsubs avoided")],
                     {"sends_suppressed": "{:.1%}", "donations_lost": "{:.1%}",
                      "first_gifts_lost": "{:.1%}", "unsubs_avoided": "{:.1%}"}), "",
            "Read: conversion barely decays with non-giving streak length, and much of it is "
            "**late first gifts** — crude `stop after k` suppression kills future donors.",
        ]))
    return sections, summary


pct_fmt = ".2%"


def render_markdown(analysis: dict, runs: dict) -> str:
    sections, summary = build_sections(analysis, runs)
    parts = [
        "# Donor platform — churn analysis & model runs",
        "",
        "Generated from `reports/analysis.json` + `reports/model_runs.json` "
        "by `experiments/render_report.py`.", "",
        "## Executive summary", "", summary, "",
    ]
    for title, lines in sections:
        parts += [f"## {title}", ""] + lines + [""]
    parts += [
        "## Method notes", "",
        "- Full population = 9.6M historical sends. The 60k frame is the repo's random send",
        "  sample with exact per-person histories (30-day engagement, send frequency, prior",
        "  donations, tenure) computed by binary search over each person's event stream.",
        "- `unsub rate` = share of sends of that group whose (person, content) has an",
        "  `unsubscribed` timeline event. Donation = transaction attributed within 7 days",
        "  (repo rule).",
        "- Policy simulation differs from `reports/baseline_report.md` in one way: it ranks",
        "  engaged sends only (non-NULL similarity), so its donations-kept numbers are higher.",
        "",
    ]
    return "\n".join(parts)


HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Donor platform — churn & model dashboard</title>
<style>
 body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0;
        background: #0f1420; color: #e8ecf4; }}
 .wrap {{ max-width: 1000px; margin: 0 auto; padding: 32px 20px 64px; }}
 h1 {{ font-size: 26px; }} h2 {{ font-size: 19px; color: #8ab4ff; margin-top: 36px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
 th, td {{ padding: 6px 10px; border-bottom: 1px solid #2a3350; text-align: left; }}
 th {{ color: #8ab4ff; font-weight: 600; }}
 .hero {{ display: flex; gap: 12px; margin: 18px 0 8px; flex-wrap: wrap; }}
 .btn {{ background: #2451ff; color: white; padding: 10px 18px; border-radius: 8px;
        text-decoration: none; font-weight: 600; }}
 .btn.alt {{ background: #233; }}
 .summary li {{ margin: 6px 0; line-height: 1.5; }}
 code {{ background: #1a2233; padding: 1px 5px; border-radius: 4px; }}
</style></head><body><div class="wrap">
<h1>Donor platform — churn analysis &amp; model runs</h1>
<div class="hero">
 <a class="btn" href="embeddings_3d.html">3D embedding evidence</a>
 <a class="btn alt" href="ANALYSIS.md">Full report (markdown)</a>
</div>
<h2>Executive summary</h2>
<ul class="summary">{summary}</ul>
{body}
<p style="margin-top:40px;color:#67718c">Rendered by experiments/render_report.py from
analysis.json + model_runs.json.</p>
</div></body></html>"""


def md_to_html_block(lines: list[str]) -> str:
    """Very small markdown-subset renderer (tables, bold, code) for the dashboard."""
    import html

    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= {"-"} and c for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>"
                                               for c in rows[0]) + "</tr>")
            for r in rows[1:]:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        t = html.escape(ln)
        t = t.replace("**", "<b>", 1).replace("**", "</b>", 1) if "**" in t else t
        t = t.replace("`", "<code>", 1).replace("`", "</code>", 1) if "`" in t else t
        if t:
            out.append(f"<p>{t}</p>")
        i += 1
    return "\n".join(out)


def main() -> None:
    analysis = json.loads((REPORTS / "analysis.json").read_text())
    runs = json.loads((REPORTS / "model_runs.json").read_text())

    md = render_markdown(analysis, runs)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ANALYSIS.md").write_text(md)

    sections, summary = build_sections(analysis, runs)
    body = "\n".join(
        f"<h2>{html_t}</h2>\n{md_to_html_block(lines)}"
        for html_t, lines in [(t, l) for t, l in sections]
    )
    summary_html = "".join(
        f"<li>{s[2:]}</li>" for s in summary.split("\n") if s.startswith("- ")
    )
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(
        HTML_SHELL.format(summary=summary_html, body=body)
    )
    (SITE / "ANALYSIS.md").write_text(md)
    print(f"wrote {REPORTS / 'ANALYSIS.md'}, {SITE / 'index.html'}, {SITE / 'ANALYSIS.md'}")


if __name__ == "__main__":
    main()
