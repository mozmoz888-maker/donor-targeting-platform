# Donor platform — churn analysis & model runs

Generated from `reports/analysis.json` + `reports/model_runs.json` by `experiments/render_report.py`.

## Executive summary

- **1.51% of sends cause an unsubscribe** (144,627 of 9,592,669 sends to 387,751 people). Donation rate is 1.15%, median gift $46.
- **Content relevance is the dominant churn lever.** Sends in similarity decile 10 (email matches the person's recent taste) unsubscribe at 1.09% vs 1.69% in decile 1 — and donate 3.51% vs 0.64% (5.5x).
- **Program area matters**: economic_empowerment churns worst (1.94%/send), hunger_relief best (1.21%).
- **Engagement protects**: no engagement in 30d = 1.55% churn vs 1.31% for the most engaged.
- Only 13% of unsubscribes even opened the email they left from — churn is background fatigue, not a reaction to content read.
- **Targeting works**: emailing the top 50% by similarity causes 43% of unsubscribes while keeping 74% of donations (vs 75%/75% for BAU). Among engaged sends only.

## Why subscribers leave: risk factors

**By program area** (full population)

| program area | sends | unsub rate / send |
| --- | --- | --- |
| economic_empowerment | 368130 | 1.94% |
| clean_water | 734580 | 1.65% |
| housing | 556321 | 1.64% |
| refugee_support | 555414 | 1.63% |
| education | 1109286 | 1.61% |
| mental_health | 924697 | 1.58% |
| maternal_health | 550075 | 1.56% |
| child_health | 923827 | 1.49% |
| medical_research | 558723 | 1.49% |
| climate_resilience | 1102151 | 1.47% |
| disaster_relief | 1286829 | 1.29% |
| hunger_relief | 922636 | 1.21% |

**By prior 30-day engagement** (60k sample, exact histories)

| engagement events | sends | unsub rate |
| --- | --- | --- |
| none | 15033 | 1.55% |
| 1-2 | 27369 | 1.49% |
| 3-5 | 15688 | 1.43% |
| 6+ | 1910 | 1.31% |

**By email frequency** (sends received in prior 30 days)

| sends in 30d | sends in sample | unsub rate |
| --- | --- | --- |
| 6-8 | 38563 | 1.51% |
| <=5 | 20473 | 1.47% |
| 9-10 | 964 | 1.04% |

**By subscriber tenure**

| tenure | sends | unsub rate |
| --- | --- | --- |
| <30d | 809283 | 1.52% |
| 30-90d | 1159624 | 1.52% |
| 180d+ | 6678714 | 1.50% |
| 90-180d | 945048 | 1.50% |

**By prior donation history**

| donated before | sends | unsub rate |
| --- | --- | --- |
| True | 11804 | 1.54% |
| False | 48196 | 1.47% |


## Embedding similarity: churn down, donations up

Each sampled send, bucketed into deciles of `centroid_cosine_similarity` (how closely the email matches the person's recent engagement):

| decile (10 = best match) | sends | unsub rate | donation rate |
| --- | --- | --- | --- |
| 2 | 4511 | 1.97% | 0.51% |
| 6 | 4497 | 1.87% | 0.98% |
| 3 | 4480 | 1.79% | 0.58% |
| 1 | 4501 | 1.69% | 0.64% |
| 4 | 4507 | 1.64% | 0.71% |
| 5 | 4485 | 1.27% | 0.78% |
| 7 | 4497 | 1.27% | 1.45% |
| 8 | 4502 | 1.16% | 1.20% |
| 10 | 4497 | 1.09% | 3.51% |
| 9 | 4490 | 0.89% | 1.87% |

**Policy simulation** (same methodology as `baseline.py`, engaged sends only):

| policy | % emailed | % of unsubscribes caused | % of donations kept | % of $ kept |
| --- | --- | --- | --- | --- |
| BAU: random 75% | 75% | 75.0% | 75.0% | 75.0% |
| centroid top 75% | 75% | 68.7% | 88.5% | 88.3% |
| centroid top 50% | 50% | 42.9% | 73.6% | 73.2% |
| centroid top 25% | 25% | 17.2% | 48.7% | 50.0% |

## Donation model (production target)

Base rate in holdout: 1.33%. Out-of-time split: holdout from 2026-08-01.

| model | ROC AUC | avg precision | kept/caused @ top 50% |
| --- | --- | --- | --- |
| logistic (baseline) | 0.619 | 0.026 | 0.00% |
| random_forest | 0.615 | 0.024 | 0.00% |
| extra_trees | 0.624 | 0.026 | 0.00% |
| hist_gradient_boosting | 0.630 | 0.020 | 0.00% |

Feature importances (random_forest): `centroid_cosine_similarity` 0.32, `tenure_days` 0.23, `days_since_eng` 0.22, `prior_amount` 0.09, `eng_events_30d` 0.06, `sends_30d` 0.06, `prior_donations` 0.02


## Churn model

Base rate in holdout: 1.34%. Out-of-time split: holdout from 2026-08-01.

| model | ROC AUC | avg precision | kept/caused @ top 50% |
| --- | --- | --- | --- |
| logistic (baseline) | 0.506 | 0.015 | 0.00% |
| random_forest | 0.547 | 0.015 | 0.00% |
| extra_trees | 0.539 | 0.015 | 0.00% |
| hist_gradient_boosting | 0.567 | 0.018 | 0.00% |

Feature importances (random_forest): `tenure_days` 0.29, `days_since_eng` 0.25, `centroid_cosine_similarity` 0.23, `prior_amount` 0.09, `sends_30d` 0.07, `eng_events_30d` 0.05, `prior_donations` 0.01


## Giving cadence: the logs have the history

Data check: 131,332 transactions from 100,110 givers ($7,609,868 total), 24% of givers gave 2+ times, 85% have at least one email-attributed gift — **the per-person giving history is fully in the logs**, nothing was stripped out.

- Repeat donors give a gift every **9 emails received** (quartiles 4/9/16). Median donor converts for the first time after **13 non-giving sends**; 60% of all first gifts arrive after 10+ non-giving sends, 33% after 20+.

- Subscriber value: 25.8% ever give, averaging 1.3 gifts of $58 (mean) in six months — an expected **$19.62 per subscriber per 6 months**. Losing one costs the mission that, plus future years.

**The non-giver ladder** (per send: how many emails since the person last gave):

| emails since last gift | sends in sample | donation rate | unsub rate | mean similarity | mean engagement |
| --- | --- | --- | --- | --- | --- |
| never gave | 48196 | 1.04% | 1.47% | 0.113 | 1.7 |
| 0-1 | 1543 | 1.17% | 1.62% | 0.128 | 3.2 |
| 2-3 | 1467 | 1.43% | 1.16% | 0.131 | 3.2 |
| 4-5 | 1316 | 1.60% | 1.82% | 0.113 | 3.0 |
| 6-10 | 2844 | 1.72% | 1.51% | 0.118 | 2.2 |
| 10+ | 4634 | 1.68% | 1.58% | 0.116 | 2.0 |

**If we stop emailing people after k non-giving emails** (sample simulation):

| k | % sends suppressed | % donations lost | % of first gifts lost | % unsubs avoided |
| --- | --- | --- | --- | --- |
| 3 | 95.0% | 94.3% | 100.0% | 95.3% |
| 5 | 92.8% | 91.3% | 100.0% | 92.6% |
| 10 | 88.0% | 84.2% | 100.0% | 87.8% |

Read: conversion barely decays with non-giving streak length, and much of it is **late first gifts** — crude `stop after k` suppression kills future donors.

## Method notes

- Full population = 9.6M historical sends. The 60k frame is the repo's random send
  sample with exact per-person histories (30-day engagement, send frequency, prior
  donations, tenure) computed by binary search over each person's event stream.
- `unsub rate` = share of sends of that group whose (person, content) has an
  `unsubscribed` timeline event. Donation = transaction attributed within 7 days
  (repo rule).
- Policy simulation differs from `reports/baseline_report.md` in one way: it ranks
  engaged sends only (non-NULL similarity), so its donations-kept numbers are higher.
