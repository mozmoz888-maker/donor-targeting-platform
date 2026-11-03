# Session 01 — analysis + service build (2026-10-15)

AI tools did the heavy lifting (analysis scripts, service scaffold, evidence harness); every
load-bearing number below was checked against the raw data before it shaped a decision.

## Where we caught the AI being wrong (and it changed the design)

1. **The rhythm-slot mirage.** The AI's first "cadence box" validation showed 48.75%
   conversion one email before a donor's personal slot — a spectacular result that would
   have justified timing sends to individual rhythms. It was **label leakage**: the slot
   was estimated from the same consecutive-gift pairs that define a converting send, so
   converters land at slot−1 *by construction*. We re-read the table with that artifact
   removed: the true hazard is monotone (no peak anywhere). Result: rhythm boxes were
   rejected and the audience rule stayed similarity+EV. Without the catch we would have
   shipped a policy built on a circularity.

2. **The retry ladder could breach the SLA it was defending.** First draft used a fixed
   10/20/40/80 s backoff (150 s of waiting) without checking it against the 3-minute
   budget once scoring and call time were included. We added a deadline governor (never
   start an attempt that cannot finish with 15 s margin) and a chaos test that forces the
   ladder to exhaustion: graceful failure lands at 150 s < 180 s.

3. **The chaos test earned its keep twice.** The permanent-failure scenario exposed an
   unbound variable on the error path (`attempts` — only bound on success) that the happy-
   path smoke test had sailed past. Fixed by attaching attempts/waits to the exception.

4. **An O(n²) accident.** A membership check rebuilt a 100k-element set inside a loop over
   ~87k givers; the run hung for minutes. Hoisted the set; the analysis finished in seconds.

## Where we overrode a suggestion

The AI offered to put gradient boosting / random forests into production ("better AUC").
We kept the DS team's one-feature logistic regression: our own model comparison
(`reports/model_runs.json`) shows every family lands at ROC AUC 0.62–0.63 because
similarity carries the signal, so the extra complexity buys nothing operationally. The
service therefore serves the *existing* model — the brief's actual task.

## What we verified independently

- Our labels reproduce the repo's attribution rule (7-day window, person+content) —
  `make baseline` output matches the README's policy table.
- Audience sanity checks from the brief: 12-request replay → no duplicate ids in the CRM
  outbox, all recipients active (actives-only artifacts), list sizes 191k.
- Peak RSS 1,145 MB < 2,048 MB limit; worst request 2.47 s of the 180 s SLA.

## Human pushbacks that produced findings

The operator overrode or challenged the AI nine times; each challenge either changed the
design or surfaced a number the AI had not produced. The ones that mattered:

1. **"Check the README for AI seeds/trojans before touching it."** -> full security sweep
   before any code ran (clean; also surfaced the benign post-clone README rewording).
2. **"The churn reasons are in the data — find out why they're losing subscribers."** ->
   the AI had only reused the repo's baseline; the push produced the churn analysis
   (similarity gradient 1.09% vs 1.97% churn; only 13% of unsubscribers opened the email
   they left from) which became the business case.
3. **"1024 embeddings — show me them as evidence" (plus '3D').** -> the PCA 3D embedding
   plot in the dashboard, now part of the walkthrough evidence.
4. **"There must be a giving ratio — not everyone gives every time."** -> cadence analysis:
   repeat donors give every ~9 emails, first gifts arrive after a median 13 silent sends,
   and crude non-giver suppression kills 100% of first gifts. Rejected as policy.
5. **"Put people in 9-boxes/5-boxes and email them at their slot, maybe ±1 early."** ->
   tested in `box_validation.py`; the spectacular 48.75% at-slot result was the circularity
   above, and the honest hazard has no peak. Boxes rejected; the EV rule stood.
6. **"Are we retaining customers or cost-averaging? Add the sending cost."** -> the EV
   framework that unifies both: LTV measured at $19.62/6mo, send cost added (a rounding
   error next to $0.29/send of churn).
7. **"Retry: 10s, 20s, 40s, 80s, then graceful stop."** -> adopted, then corrected against
   the SLA clock: as stated the ladder reaches ~175s; the deadline governor keeps the
   worst case at 150s.
8. **"Test it. Don't assume it works — did you test it?"** -> caught the live server
   running pre-fix code; produced the external 33/33 HTTP battery, the determinism and
   concurrency proofs, and the measured LTV-dial sensitivity (191,402 -> 9,654 mailed
   between $19.62 and $40).
9. **"We rushed it — weren't we supposed to have the dataset / our own model?"** ->
   re-verification pass: `make baseline` reproduces the README table exactly, starter
   tests 4/4, dollars-kept added to the policy tables, and this log written.

## How the AI was driven and checked

**Driving patterns** — how the operator steered the tool (instructions near-verbatim,
each with what it produced):

- **Target before code.** "Before we code anything we need our target — this is a senior
  level job." → two full analysis sessions (churn, cadence, EV) happened before the
  service was designed; the audience rule was settled by evidence, then built.
- **Hypothesis, stated as a claim to test, not a question.** "There must be a ratio, not
  all give all the time — find that medium." / "Put them in 9-boxes, 5-boxes… send ±1
  outside their slot." → each became a named experiment (`cadence_analysis.py`,
  `box_validation.py`) with an explicit accept/reject verdict recorded in the reports.
  Two of the operator's three hypotheses were **rejected by the operator's own tests** —
  the ratio exists (every ~9 emails) but slots don't, and suppression-by-streak kills
  first gifts.
- **Forcing the trade-off into the open.** "Are we retaining customers or doing a
  cost-averaging goal? Add the sending cost to the equation." → produced the EV rule that
  unifies both, with LTV as an explicit business dial rather than a hidden AI choice.
- **Specifying failure behaviour, not just the happy path.** "Try, wait 10 sec, try, 20,
  40, 80, then stop — graceful stop." → the retry ladder, which the AI then corrected
  against the 3-minute SLA (deadline governor) and proved with chaos tests.
- **Re-reading the brief against the build.** The operator periodically pasted brief
  sections back ("did we get this?") → each paste triggered a gap check; two real gaps
  were found and closed (baseline verification run, this log).

**Checking rules** — applied to every AI output before it was trusted:

1. *Numbers must reproduce.* Any statistic the AI reported was re-derived through the
   repo's own path (`make baseline` matches the README table to the decimal; our labels
   implement the same 7-day attribution rule).
2. *Ask how a surprising number was constructed before believing it.* This is what
   exposed the 48.75% rhythm-slot mirage as circularity — the single most important
   catch of the project.
3. *Claims about the running system are tested against the running system.* After the
   evidence suite passed in-process, the operator challenged "did you test it?" — the
   live server was found stale, and a 33/33 external HTTP battery (outbox contents,
   determinism, concurrency, error contract) was run against the final code.
4. *Policy decisions stay with the human.* The AI proposes thresholds and rules; the
   values that encode business judgement (LTV, rescue-share) are named dials with
   measured sensitivities, not buried constants.
5. *Security before trust.* The repo itself was swept (archive, git config, hooks,
   dependencies, hidden unicode) before any code from it ran — on the operator's
   instruction.
