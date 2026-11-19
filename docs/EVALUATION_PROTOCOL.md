# Evaluation protocol: how to find out whether the firm has skill

This document turns the staged gates of the research report
([RESEARCH 7.4](research/RESEARCH_REPORT.md)) into steps you can run with this code. For
each stage it gives what the research asks, the commands, the report fields to read, the
pass thresholds the code applies, and what the code does **not** check for you. The
design itself is in [ARCHITECTURE.md](ARCHITECTURE.md).

Read the stages in order. A later stage means nothing if an earlier one failed.

## Terms used below

* **Contaminated.** A decision made on a date the model may have seen in training. Its
  "forecast" can be memory. Here a decision is contaminated when it is on or before
  `contamination_end` = the model's training cutoff plus `evaluation.cutoff_buffer_days`
  (31 calendar days).
* **Clean window.** The dates after `contamination_end`.
* **Rank IC** (information coefficient). On one decision date, the Spearman rank
  correlation between the probabilities a signal gave and the realised excess returns.
  The mean over dates, with its t-statistic, measures ranking skill. The date is the unit
  of replication.
* **Newey-West t-statistic.** A t-statistic whose standard error allows for overlapping
  forecast windows (used when decisions are closer together than the horizon).
* **Paired IC test.** The per-date difference between two signals' ICs on the same dates,
  and its t-statistic. It is the test for "does design A rank stocks better than B".
* **PSR, DSR, MinTRL.** Probabilistic Sharpe ratio (probability the true Sharpe is above
  zero), deflated Sharpe ratio (PSR against the best Sharpe expected from the number of
  configurations tried), and minimum track record length (how long a record must be for
  95% confidence that the true Sharpe is above zero).
* **Composition-matched random books.** Random portfolios that copy the strategy's
  weights, number of names, cash, core and holding periods, but hold names drawn at
  random. They are the null for stock selection.
* **Breakeven multiplier.** The factor by which every modelled cost must be multiplied
  for the strategy's excess return to reach zero.
* **Trial.** One distinct (config fingerprint, mode, start, end) in
  `<runs_dir>/trials.jsonl`. The trial count feeds the DSR.

## Summary

| Stage | What it establishes | How to run it here | Status |
|---|---|---|---|
| S0 | The machinery is correct, without any LLM | `pytest`; `aifirm backtest --mode quant` and `--mode offline`; `aifirm verify-ledger` | Partly automated |
| S1 | Each model's contaminated period | `MODEL_TRAINING_CUTOFFS` plus the 31-day buffer, shown in line 2 of the six lines | Partly automated (vendor dates only) |
| S2 | Blind mode really hides the stock and month | `aifirm reid-probe` | Automated gate; not enforced by the backtest |
| S3 | Development on pre-cutoff data does not tune prompts on returns | the trial ledger; your discipline | Not automated |
| S4 | One design beats another on several model vintages | backtest report: paired IC, DSR, factor alpha, random books | Per run automated; the ladder is not |
| S5 | One untouched holdout | one `aifirm backtest --mode llm` on the clean window | Not automated |
| S6 | Forward skill, net of LLM cost | `aifirm run` each cycle, `aifirm compare` | Automated measurement and verdict; capital steps are human |

## S0. Machinery, no LLM

**The research asks** that the data layer, costs, risk engine and statistics be validated
with stub agents: planted leaks of at least 4 types, 12 instances each, all caught; a
replay with the future randomised after cuts on decision days leaves earlier decisions
unchanged; ledgers replay exactly; a planted IC of 0.02 to 0.03 is recovered; no limit is
breached on stress paths; results survive 2× costs.

**Commands.**

```bash
pytest                                                    # offline suite, synthetic data, no API key
aifirm backtest --config my.toml --start 2022-01-01 --end 2025-12-31 --mode quant
aifirm backtest --config my.toml --start 2022-01-01 --end 2025-12-31 --mode offline
aifirm verify-ledger --config my.toml                     # add --offline or --shadow for the other books
```

Quant mode uses no language model. Offline mode replaces every LLM with deterministic
rules (not AI); it exercises every stage of the committee.

| Requirement | Where it is checked | Status |
|---|---|---|
| Planted leaks caught | Unit tests per leak type: future prices scrambled after a cut (`tests/test_backtest.py::test_backtest_has_no_look_ahead`), a filing accepted after the 16:00 close is not known that day, macro prints hidden until released, partial intraday bars dropped, headlines after the close dropped, lessons, forecasts, believability, calibration, the overlay bound and the screen all point-in-time | **Partly.** Several leak types, each with one or a few instances; there is no 4 × 12 planted-leak suite. |
| Future randomised after cuts | `tests/test_backtest.py::test_backtest_has_no_look_ahead` (one cut: NAV and decisions up to the cut unchanged); `tests/test_pipeline_wiring.py::test_backtest_with_a_screen_is_point_in_time_and_records_the_deep_dive` | **Partly.** One cut per test, not a cut on every decision day. |
| Ledgers replay exactly | `aifirm verify-ledger` (hash chain plus the recorded head); tamper tests in `tests/test_ledger.py`; every backtest re-executes its recorded targets at 1× costs and checks that the NAV matches the engine (`replay_matches_engine`; a mismatch is printed in the report) | **Partly.** No tool re-runs a recorded live cycle from its ledger. |
| Planted IC recovered | `tests/test_evaluation.py::test_a_planted_ic_of_about_005_is_recovered_with_a_positive_t_stat`: 100 names × 120 dates, planted IC about 0.05, requires a mean IC between 0.03 and 0.08 and t > 3; a pure-noise signal must have \|t\| < 2.5 | **Partly.** The planted IC is larger than the 0.02 to 0.03 the research asks for. |
| No limit breaches on stress paths | Risk-engine and construction tests (hard cap, sector cap, turnover, freeze in a synthetic crash, de-risk hysteresis); the risk engine only moves weights toward zero | **Partly.** No dedicated stress-path harness. |
| Survives 2× costs | Every backtest report, "Cost sensitivity and breakeven": the same decisions re-executed at k = 0, 1, 2 (`evaluation.cost_multipliers`) and the breakeven multiplier | **Automated.** |

**Pass, as the code reports it.** The suite passes. The backtest report's evaluation notes
show no failed part and no replay mismatch. In the cost table, the excess return over the
benchmark at 2× is above zero, which is the same as a breakeven multiplier of at least 2
(status `found` with a multiplier ≥ 2, or `above_cap`). The interpretation says so in words
when the result does not survive 2×. At S0 the point is that these numbers are computed
correctly; a quant or offline strategy is not expected to pass them.

## S1. Contamination audit per model

**The research asks** for each model's measured cutoff, its clean window and a pinned
serving configuration. Vendor cutoffs differ from measured ones by up to 8 months, so the
clean window should start at the later of the two dates plus one month (RESEARCH 7.2).

**What the code does.**

* `aifirm.config.MODEL_TRAINING_CUTOFFS` holds the **vendor** cutoff per model
  (`claude-opus-5-5`: 2026-06-30). A model that is not listed counts as cut off today, so
  every past decision is contaminated.
* `contamination_end` = the latest cutoff among the run's models plus
  `evaluation.cutoff_buffer_days` (31). For `claude-opus-5-5` that is 2026-07-31.
* The backtest report states it in line 2 of the six lines ("decisions after ... are clean"), says
  where the test lies ("the whole test lies after it (clean)", "straddles it" or "lies
  before it"), shows a contamination warning, and splits returns into pre- and
  post-cutoff tables. `aifirm backtest` prints the same date.
* Each forecast records the model that actually served it. With
  `models.use_server_fallbacks = true` (the default), a refused call may be answered by a
  different model. For a controlled study, set it to false so every call uses the pinned
  model.

**Not automated.** Measuring a model's effective cutoff. If you measure one that is later
than the vendor date, put the later date in `MODEL_TRAINING_CUTOFFS` by hand (a code
change). The `pre_cutoff` flag on forecast records uses the raw cutoff without the 31-day
buffer, so the scorecard's "pre-cutoff" column ends 31 days earlier than the backtest split.

**Pass.** Every decision you count as evidence lies after `contamination_end`.

## S2. Re-identification gate

**The research asks** that, before any backtest on pre-cutoff data, the exact payload the
model will see is probed at least 400 times, and joint re-identification (ticker and date)
is no more than 1 percentage point above chance with a 95% upper bound of at most 2%
(RESEARCH 7.2, 9.3 item 2).

**When you need it.** Before any LLM backtest with decisions on or before
`contamination_end`. Such a backtest must also run with `process.blind_mode = true`
(blind mode is off by default). A backtest entirely in the clean window does not need it.

**Commands.**

```bash
# test the harness first: a seeded random guesser, no credentials, no cost
aifirm reid-probe --config my.toml --start 2024-01-02 --end 2026-05-29 --n 400 --offline
# the real probe: 400 calls to the model; --budget sets the dollar cap
aifirm reid-probe --config my.toml --start 2024-01-02 --end 2026-05-29 --n 400 --budget 20
aifirm reid-probe --config my.toml --start 2024-01-02 --end 2026-05-29 --n 400 --role technical
```

Use the same universe and window as the backtest you want to run. The probe samples
(date, ticker) pairs from the rebalance dates and builds historical fact sheets (no news).
The model sees the full blind sheet (the default; a superset of every analyst slice) or
one analyst's slice (`--role`), the whole universe as a candidate list and the month
window. The default model is `models.committee_model`.

**Output.** `rep.summary()` on the console; `<runs_dir>/reid/reid_<time>_<model>.json`;
a `reid_probe` entry in `<runs_dir>/reid/ledger.jsonl`.

**Pass (the gate, as coded).** All of:

* at least 400 **answered** probes (a failed call, including one refused for budget, is
  excluded, never counted as a miss);
* joint hit rate minus joint chance ≤ 0.01;
* joint Wilson 95% upper bound minus joint chance ≤ 0.02;
* no payload contains the true ticker, the as-of date or the company name verbatim.

The command exits with status 0 when the gate passes and 1 when it fails. Because the
probe is closed-world (a candidate list), both thresholds are applied to the excess over
chance; joint chance is 1/(candidates × months).

**Not automated.** The backtest does not check for a passing report before a blind
pre-cutoff LLM run: you must. **Known weakness:** blind mode keeps the SEC SIC sector
label (scrubbed of the ticker and name). On real data it can nearly identify a company
("Electronic Computers"). Expect it to cost re-identification power and check the ticker
hit rate as well as the joint one.

## S3. Pre-cutoff development

**The research asks** that pre-cutoff data be used to tune only plumbing, schemas, risk
and costs, never prompts or debate settings of text-reading agents against pre-cutoff
returns.

**What the code gives you.**

* Quant and offline backtests need no model and cannot memorise anything.
* Every backtest appends a row to `<runs_dir>/trials.jsonl`. The number of distinct
  (fingerprint, mode, start, end) trials raises the DSR bar.
* Strategy flags (`--preset`, `--combination`, `--max-tilt`, `--no-calibration` and so
  on) change the fingerprint, so each counts as a trial. Evaluation flags do not.

**Not automated.** Nothing stops you from tuning prompts on pre-cutoff returns. Prompts
are code, not configuration, so a prompt edit does **not** change the fingerprint and is
not counted as a trial. If you edit prompts, change the config's `name` (it is part of the
fingerprint, for example `name = "firm prompts v2"`) so the trial ledger counts the
variant. Keep `trials.jsonl`: deleting it makes every DSR look better than it is.

## S4. Architecture selection on a vintage ladder

**The research asks** that each candidate design be tested on at least three models with
staggered cutoffs, each on its own clean window, with at least three seeds; designs are
compared with paired per-date IC-difference tests; the simpler design wins unless the
difference reaches t ≥ 2 with a consistent sign; IC t ≥ 3 is required if the pick came
from more than 20 trials; DSR ≥ 0.95 and PBO ≤ 0.2; factor alpha t ≥ 2; and the 95th
percentile against composition-matched random books. There is no published evidence that
rankings transfer from older models to newer ones; the ladder is a proxy.

**Commands** (one design, one model, one clean window):

```bash
aifirm backtest --config my.toml --start 2026-08-03 --end 2026-11-13 --mode llm --preset sp100
aifirm backtest --config my.toml --start 2026-08-03 --end 2026-11-13 --mode llm --preset sp100 --combination pool
```

These dates lie after `contamination_end` for `claude-opus-5-5` (2026-07-31). A window this
short holds only a few decision dates whose forecasts resolve inside it, so the report
will call the result anecdotal; it shows the commands, not a sufficient test.

**What each backtest report computes** (section "Is there skill, and would we know?" and
the six lines):

| Research criterion | Report field | Threshold the code applies |
|---|---|---|
| Paired IC test against the simpler design | Paired table: "Committee pool minus Quant-only baseline" (mean difference, t-stat, reading). The interpretation repeats it. | Reading "positive (t >= 2)"; otherwise "default to the simpler design" |
| Weighted versus equal-weight pool | Paired table: weighted minus equal-weight pool | t ≥ 2 |
| IC significance, corrected for trials | IC table, pool row; the interpretation | t ≥ 2, or t ≥ 3 when more than 20 trials are recorded |
| DSR ≥ 0.95 | Line 5 of the six lines; "Statistical significance" | DSR ≥ 0.95 (not computed with a single trial: PSR is shown instead) |
| Factor alpha t ≥ 2 | "Factor attribution (ETF proxies)": alpha and its HAC t-stat | t ≥ 2 |
| 95th percentile against random books | Line 6 of the six lines; random-books table | percentile ≥ 95 ("its stock selection beat random picks") |
| Value of the LLM stages net of their cost | "Value added by the language models" | above zero (no significance test; use the paired IC) |
| Power | "Power: how long until we would know" | "Anecdotal" unless the sample has the dates needed for 80% power |

In overlay mode the "Committee pool" signal is the firm's final probability (the quant
core plus the bounded LLM tilt), not a pure LLM pool, although the report labels it
"believability-weighted" in both modes. Names outside the deep dive carry the quant's
probability in both runs, so the paired test against the quant-only baseline measures
what the LLM stages changed in the ranking.

**Not automated.**

* **The ladder.** Only one model has a cutoff in `MODEL_TRAINING_CUTOFFS` and there is no
  loop over models. Run each model yourself (set `models.analyst_models`,
  `committee_model` and `risk_model`, add its cutoff) on its own clean window.
* **Seeds.** Identical calls are answered from the disk cache, so re-running the same
  backtest returns the same answers. `models.samples_per_analyst` adds samples inside a
  run; it is not a seed of the whole run.
* **Comparing two LLM designs from separate runs.** The paired test in the report is only
  against the quant-only baseline and the equal-weight pool. The CLI saves the report, not
  the result object, so for design A against design B run both backtests from Python
  (`aifirm.backtest.engine.run_backtest`), take `result.evaluation.ic["pool"]` from each and
  call `aifirm.backtest.ic.paired_ic_difference(a, b)`. Use the same window, universe, screen
  and rebalance dates. Check the sign across models by hand.
* **PBO** (probability of backtest overfitting) is not computed.
* **Results by regime** are not reported.

## S5. One-use holdout

**The research asks** for one evaluation, used once, of the production model on the window
between its cutoff and its release.

**Command.** One `aifirm backtest --mode llm` on that window, with the configuration fixed
beforehand.

**Not automated.** Nothing stops a second run. The trial ledger records every run with its
window, so a repeated holdout is visible in `trials.jsonl` and raises the trial count.

## S6. Forward paper trading

**The research asks** for forward paper trading of one frozen configuration: at least 126
trading days plus IC t ≥ 2.5 for broad daily books; at least two years for books of 100
names or fewer, or for weekly or monthly horizons; then pilot capital of at most 10% of
the target; and scaling only after the minimum track record is met with every watched
variant deflated.

**Commands** (after 16:30 New York time on a decision day, typically monthly):

```bash
aifirm run --config my.toml                  # live paper book + quant-only shadow book
aifirm compare --config my.toml              # AI value added net of LLM cost, paired IC, verdict
aifirm score --config my.toml                # scorecard; paired IC: live vs shadow, LLM pool vs quant
aifirm verify-ledger --config my.toml        # and again with --shadow
```

**What the code measures.**

* `AI value added net of LLM cost = (live NAV − shadow NAV) − LLM spend since inception`,
  at the latest close both books decided.
* The paired per-date rank IC of the live final probability against the shadow's, on the
  names both forecast, from forecasts resolved by the as-of date (Newey-West when
  decisions are closer than the horizon).

**Verdict thresholds (as coded in `shadow.ic_verdict`).**

* **Not yet measurable** until there are at least `max(process.overlay_min_dates,
  ceil(2 × 252 / step))` resolved decision dates, where `step` is the median gap between
  live decisions: 24 dates at a 21-trading-day cadence, 504 for daily runs.
* **Better**: then t ≥ 2 and a positive mean difference.
* **Worse**: t ≤ −2. The overlay should be shrunk or switched off.
* **No difference** otherwise: default to the simpler quant-only design.

**Differences from the research and what is not automated.**

* The code applies the two-year minimum and t ≥ 2 to every book. The 126-day minimum with
  IC t ≥ 2.5 for broad daily books is not implemented.
* "One frozen configuration" is not enforced. Every `cycle_start` ledger entry records the
  config fingerprint; check that it never changed during the evaluation.
* The adaptive overlay bound is driven by the LLM pool versus the quant, not by the shadow
  comparison or by net-of-cost value.
* Pilot capital, scaling, a live MinTRL and deflating watched variants are human steps.
  Moving to real money is outside this software (see the README's safety notes).
* With an Alpaca account, broker NAV can include deposits and withdrawals the shadow does
  not see; the NAV comparison is only clean for paper books. The IC comparison is not
  affected.

## The six-line report standard

Every backtest report opens with six lines (RESEARCH 7.4, after arXiv 2611.18934), and
`aifirm backtest` prints them to the console. `aifirm.report.six_lines(result)` builds them:

1. **Window and assets.** Dates, trading days and years; number of stocks plus the
   benchmark; a survivorship flag when the universe list is today's membership.
2. **Model and cutoff.** The models used, the training cutoff plus the buffer, and where
   the test lies (clean, straddling or contaminated). In quant mode: none.
3. **Cost model and breakeven.** Commission, spread and square-root impact; the realised
   cost in basis points per dollar traded; the breakeven multiplier against the benchmark.
4. **Configurations tried.** Distinct trials in `trials.jsonl`.
5. **Deflated statistic.** The DSR at that trial count against the 0.95 bar; with a single
   trial, PSR and Sharpe instead.
6. **Benchmark and null.** Excess total return over the benchmark, and the percentile and
   one-sided p-value against composition-matched random books.

The research also asks for factor attribution (reported), results by regime (not
reported) and an equal-weight-pool versus weighted-pool comparison (reported as a paired
IC test).

## The red-team suite

**The research asks** (RESEARCH 7.4, 9.3 item 7) for injection, fake-news, fake-tool and
position-poisoning payloads and renamed real hoaxes, run in CI in a sandbox, and scored per
decision on attack success, turnover multiple, maximum concentration and orders reaching
the broker (zero for blocked classes), not on P&L.

**What exists** (all offline; run them with `pytest`):

| File | What it asserts |
|---|---|
| `tests/test_news_quarantine.py` | Red-team fixtures: five injection headlines, two fake-news stories from unknown publishers, a spoofed "Reuters" link and a genuine Business Wire release. With an *obedient* (compromised) reader, tiers still come from code and summaries are sanitised. No raw headline text, nor any four-word window of it, reaches any analyst, challenge, revision, memo or risk-review prompt. A credulous analyst's bullish view resting only on tier C news is capped at 0.5. A genuine tier A event does support a bullish view. |
| `tests/test_news_reader.py` | The 16:00 New York cutoff; a source-tier table including spoofs (look-alike domains, subdomain tricks, userinfo `@` tricks, homoglyphs, contradictory attribution, missing or `javascript:` links); the injection detector on known payloads and on ordinary headlines; prompt break-out neutralised; trusted and untrusted headlines read in separate calls; fail-closed on every reader failure. |
| `tests/test_tier_gate.py` | What counts as admissible support; the gate never raises a probability. |
| `tests/test_ledger.py` | Edited, re-hashed, deleted, reordered, whitespace-edited and inserted lines are detected; a deleted tail is caught by the recorded head. |
| `tests/test_reid.py` | A simulated blinding bug (identity in the payload) fails the gate; a memorising model fails it. |
| `tests/test_pipeline_e2e.py`, `tests/test_shadow.py` | The kill-switch file blocks every order, for the live and the shadow book; a failing shadow book cannot change the live decision. |

**Not implemented.** A scoring harness that runs attack classes through full decisions
and counts attack success, turnover multiples, concentration and orders reaching the
broker; TradeTrap payloads and renamed real hoaxes; position poisoning of broker state
(a corrupt state file stops the run with an error, which is the only defence). Fake-tool
(MCP) payloads do not apply: no agent has tools. The injection detector is a set of
regular expressions; the real defences are code-assigned tiers, separate reader calls,
sanitised summaries and the tier gate.

## What passing does not mean

Passing every automated check is necessary, not sufficient. A monthly book of about ten
names cannot statistically confirm skill on any realistic horizon (RESEARCH 7.3); use the
screen for breadth or accept that the answer takes years. The bundled `sp100` list is
today's membership, so its backtests are biased upward. Only forward, post-cutoff,
after-cost evidence counts.
