# Baseline: BAU vs centroid targeting

60,000 sampled sends from 2026-03-31 to 2026-08-20 (42 emails): 690 donations, 891 unsubscribes.

| policy | emailed | precision | recall | $ recall | unsub share |
| --- | ---: | ---: | ---: | ---: | ---: |
| send to everyone | 100.0% | 1.15% | 100.0% | 100.0% | 100.0% |
| BAU: random 75% | 75.0% | 1.15% | 75.0% | 75.0% | 75.0% |
| engaged in last 30 days | 74.9% | 1.22% | 79.7% | 79.6% | 73.8% |
| centroid top 75% | 75.0% | 1.24% | 81.0% | 81.2% | 73.6% |
| centroid top 50% | 50.0% | 1.51% | 65.8% | 63.7% | 43.1% |
| centroid top 25% | 25.0% | 2.18% | 47.2% | 46.0% | 19.8% |

Precision = donation rate of the people emailed. Recall = share of all donors still reached. Unsub share = share of all unsubscribes still caused.
