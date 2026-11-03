# Service evidence

Peak RSS: **1145.2 MB** of the 2048 MB limit. Worst request→CRM: **2.5s** of the 180s SLA.

| scenario | request | mailed | suppressed | score_s | total_s |
| --- | --- | ---: | ---: | ---: | ---: |
| A normal | req_2001 | 191,444 | 52,613 | 0.329 | 2.44 |
| A normal | req_2002 | 191,476 | 52,581 | 0.304 | 2.41 |
| A normal | req_2003 | 191,441 | 52,616 | 0.316 | 2.42 |
| A normal | req_2004 | 191,384 | 52,673 | 0.308 | 2.41 |
| A normal | req_2005 | 191,424 | 52,633 | 0.311 | 2.42 |
| A normal | req_2006 | 191,414 | 52,643 | 0.308 | 2.41 |
| A normal | req_2007 | 191,406 | 52,651 | 0.308 | 2.41 |
| A normal | req_2008 | 191,281 | 52,776 | 0.363 | 2.47 |
| A normal | req_2009 | 191,394 | 52,663 | 0.312 | 2.42 |
| A normal | req_2010 | 191,425 | 52,632 | 0.329 | 2.44 |
| A normal | req_2011 | 191,357 | 52,700 | 0.316 | 2.42 |
| A normal | req_2012 | 191,402 | 52,655 | 0.32 | 2.42 |
| B: CRM fails twice then recovers | | | | | 32.5 |
| C: CRM permanently down | | | | | 150.5 |

## Assertions

- PASS: 12/12 requests sent, one CRM list each, no duplicate person ids
- PASS: worst request→CRM wall time 2.5s < 180s SLA
- PASS: transient CRM failure recovered: attempts=3, waits 10s+20s, done in 32s
- PASS: permanent CRM failure: graceful FAILED in 150s (< SLA), audience persisted=True, status endpoint shows terminal state
- PASS: duplicate request_id -> 409 with prior list_id (no second CRM list)
- PASS: bad program_area / wrong embedding length -> 422 fail-fast
- PASS: peak RSS 1145.2 MB < 2048 MB worker limit
