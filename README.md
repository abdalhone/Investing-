# aifirm: a research and paper-trading system for multi-agent stock selection

`aifirm` runs a small investment firm made of software agents, on paper. A data desk
builds point-in-time fact sheets in which every number has an ID and a source. A
deterministic quant factor model scores every stock. Several Claude analysts each read a
different slice of the facts and write sealed, independent forecasts of whether a stock
will beat the benchmark over the next month. Code pools those forecasts and, by default,
lets them tilt the quant model's probability only a little. A committee chair audits the
result and may only shrink it. Deterministic code sizes positions, enforces risk limits
and sends orders to a paper broker. Every step goes into a tamper-evident ledger, every
forecast is scored when its horizon passes, and a quant-only **shadow book** runs beside
the live book so you can see whether the language models earn what they cost.

```
Data desk (Yahoo prices, SEC filings as filed, FRED macro; every number has an ID)
   -> quant factor model scores the universe -> optional screen: LLM deep dive on the top band + held names
   -> raw headlines -> quarantined news reader -> typed events, source tier A/B/C set by code
   -> analysts: fundamental | technical | macro | events (Claude) + quant model (code), sealed views
   -> tier gate (unverified news cannot lift a view) -> calibration per analyst
   -> LLM pool tilts the quant probability by at most 0.05 (bounded overlay)
   -> challenge round only if members disagree -> private revisions
   -> IC memo by the chair (can only shrink the edge)
   -> portfolio engine (code): quarter Kelly, 8% cap, sectors, volatility target, half-way rebalancing
   -> LLM risk review (can only shrink) -> risk engine (code: drawdown rules, loss halt, kill switch)
   -> orders -> paper broker (fills at the next open, with costs) -> hash-chained ledger
   -> forecasts resolve -> Brier, rank IC -> weights, calibration, overlay bound, lessons (point-in-time)
   -> quant-only shadow book at the same close -> `aifirm compare`
```

The design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), the evidence in
[docs/research/RESEARCH_REPORT.md](docs/research/RESEARCH_REPORT.md), and the way to test
for skill in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md). In one line:
**LLMs propose, code disposes.** Agents output probabilities and evidence. They never
choose weights, share counts or leverage.

## Honest expectations

- **There is no audited, post-cutoff, after-cost evidence that LLM stock picking adds
  alpha.** As of November 2026 the striking published results do not survive careful
  re-testing: long horizons, testing after the model's training cutoff and
  better-controlled benchmarks all erase the edge (research report, section 1). This is a
  research and paper-trading system built to measure that honestly, not to promise
  returns.
- **A book of about ten names decided monthly cannot statistically validate skill.** At a
  realistic rank IC it would take decades to tell skill from luck (research report 7.3).
  Breadth helps: the optional screen ranks a broad universe so the forecasts can be judged
  on many names per date.
- **The quant-only shadow book is how you find out whether the AI earns its cost.** Every
  `aifirm run` also decides a paper book with every language-model stage removed.
  `aifirm compare` reports the difference in NAV minus the LLM spend, and a paired test of
  whether the AI ranks stocks better. It says "not yet measurable" until about two years
  of monthly decisions have resolved.
- **Good one-month forecasts sit close to 0.50.** The quant model's probabilities stay
  between 0.44 and 0.56, the LLMs can move them by at most 0.05, and a name needs 0.52 to
  be held. On a narrow universe the book is therefore often mostly cash; every report
  shows how much was invested and why. Set `portfolio.core_weight` (for example 0.5) if
  you want the rest to track the benchmark.
- **Backtests before a model's training cutoff are contaminated.** The model may remember
  what happened next. Reports treat decisions up to the cutoff plus 31 days as
  contaminated and split them out. Only later decisions, and forward paper trading, can
  show skill.
- **The offline mode is not AI.** It replaces every LLM with simple deterministic rules so
  you can run and test the whole machine without an API key. Never act on its output.

A development run shows what an honest report looks like. A quant-only backtest (no LLM)
from 2022-01-01 to 2025-12-31 on the bundled `sp100` list returned +37.1%, against +50.9%
for SPY and +51.3% for an equal-weight basket of the same names. Its forecasts had no
ranking skill (mean rank IC −0.019, t = −0.63 over 47 decision dates), it sat at the 62nd
percentile of 1,000 composition-matched random books (p = 0.38), and its factor-adjusted
alpha was +1.5% a year with t = 0.48. The list is today's index membership, so even those
returns are biased upward. The report said all of this in its first six lines and its
"Is there skill, and would we know?" section.

## Quick start

Requires Python 3.11 or later (the config loader uses `tomllib`).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC asks automated clients to identify themselves
pytest                                                      # the test suite runs offline on synthetic data
```

`python -m aifirm ...` works the same as the `aifirm` command. A relative `runs_dir` is
resolved against the directory of the config file, not the directory you run from: with
`examples/firm.toml` (`runs_dir = "runs"`) the books live in `examples/runs/`. Every run
prints the book path and the kill-switch path it uses.

### 1. Look at a fact sheet

```bash
aifirm factsheet AAPL                    # the latest completed session
aifirm factsheet AAPL --as-of 2026-06-30
```

This is the full sheet; each analyst sees only its own slice of it. For the latest
session it also lists raw headlines, for you to read: the agents never see raw headlines.

### 2. Screen a broad universe

The screen is off in `examples/firm.toml`. To screen the bundled S&P 100 list, make a
config such as `my.toml` (every other setting keeps its default):

```toml
runs_dir = "runs"

[universe.screen]
enabled = true
preset = "sp100"          # today's members: survivorship-biased in backtests
```

```bash
aifirm screen --config my.toml           # ranking, ineligible names and the deep-dive set; no LLM, no orders
```

`--top N` changes the deep-dive size, `--rows N` shortens the list and `--record` writes
the ranking to the book's ledger. On `run` and `backtest`, `--preset sp100`,
`--universe-file PATH`, `--deep-dive N` or `--min-dollar-volume X` turn the screen on
without editing the config.

### 3. Run one cycle offline (no API key)

```bash
aifirm run --config examples/firm.toml --offline
aifirm compare --config examples/firm.toml --offline
```

The offline book lives in `<runs_dir>/offline/` and its shadow in
`<runs_dir>/offline/shadow/`, apart from the real paper book.

### 4. Backtest

```bash
aifirm backtest --config my.toml --start 2022-01-01 --end 2025-12-31 --mode quant     # no LLM
aifirm backtest --config my.toml --start 2025-01-02 --end 2026-10-30 --mode offline   # plumbing check
aifirm backtest --config my.toml --start 2026-08-03 --end 2026-11-13 --mode llm       # Claude; costs money
```

Every backtest decides at the close and fills at the next open with costs, marks the book
every day, credits dividends, resolves forecasts as time passes and uses only outcomes
known at each decision date. It runs three baselines through the same engine: the
benchmark bought and held, an equal-weight universe with the same dates and costs, and
(except in quant mode) the quant model alone. It then evaluates the run: signal IC and
power, composition-matched random books, the same decisions re-run at 0×, 1× and 2× costs
with a breakeven, and a factor regression on ETF proxies (the CLI downloads them). The
report is written to `<runs_dir>/backtests/<run_id>/report.md`; the console prints its six
summary lines.

Every run is appended to `<runs_dir>/trials.jsonl`. The number of distinct configurations
and windows you have tried raises the bar of the deflated Sharpe ratio, so keep that file.
Strategy flags (`--combination`, `--max-tilt`, `--no-calibration`, `--no-news-reader`,
`--no-tier-gate`, `--rebalance-fraction`, the screen flags) count as new configurations.
Evaluation flags (`--random-books`, `--seed`, `--sector-matched-books`,
`--cost-multipliers`, `--breakeven-cap`, `--cutoff-buffer-days`, `--ic-min-names`,
`--no-factor-proxies`) do not.

### 5. Test blind mode before any pre-cutoff LLM backtest

```bash
aifirm reid-probe --config my.toml --start 2024-01-02 --end 2026-05-29 --n 400 --offline   # harness check
aifirm reid-probe --config my.toml --start 2024-01-02 --end 2026-05-29 --n 400 --budget 20 # the real gate
```

The probe shows the model 400 blind fact sheets and asks it to name the stock and month.
The gate passes only with at least 400 answered probes, a joint hit rate (stock and month)
no more than 1 percentage point above chance, a 95% upper bound no more than 2 points above
chance, and no sheet that contains the ticker, date or company name. The command exits
with status 1 when the gate fails. An LLM backtest with decisions on or before the model's
cutoff plus 31 days needs `process.blind_mode = true` and a passing probe on the same
universe and window; the backtest does not check this for you. See
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), stage S2.

### 6. Live paper trading with Claude

```bash
export ANTHROPIC_API_KEY=...        # or run `ant auth login` once to create a profile
aifirm run --config examples/firm.toml
aifirm compare --config examples/firm.toml
```

Run it once per decision cycle, after 16:30 New York time (for example monthly). Each run:

1. settles the paper book since the previous run: credits cash dividends on each ex-date
   to the shares held at the previous close, and fills the orders queued by the previous
   run at the **next trading day's open** after that decision, with commission, spread and
   market impact;
2. resolves forecasts whose horizon has passed and writes a few process lessons;
3. screens (if enabled), reads the headlines of the names under review through the
   quarantined news reader, runs the committee for each of those names, sizes the book,
   applies the risk review and the risk engine, and queues new orders for the next open;
4. decides the quant-only shadow book at the same close (unless `--no-shadow`, or the
   configuration has no LLM stage);
5. writes the report to `<runs_dir>/reports/decision_<date>_<run_id>.md`, appends every
   step to `<runs_dir>/ledger.jsonl`, and prints warnings, every rule the risk engine
   applied, any order it blocked, the exposure and the shadow comparison.

A run refuses to decide again before the queued orders have filled, and a book never
decides on a session it has already decided or on an earlier one (`--force` overrides the
second rule and is written to the ledger). Only completed sessions are used: `--as-of`
may name an earlier session but not today before 16:30 or a future date. A holiday or
weekend means the session before it.

**What a run costs.** The hard cap is `process.max_llm_cost_usd_per_run` (default $10).
The run stops calling the model once spending reaches it (it can overshoot by at most one
call), and an exhausted budget holds the book rather than trading on partial views. With
the default 10 names and `claude-opus-5-5` ($4 per million input tokens, $20 per million
output tokens; check current prices), a cycle makes about 40 analyst calls, memos for held
names and names with an edge, occasional challenge rounds, one risk review, up to 8
lessons and up to two low-effort news-reader calls per name with headlines. An earlier
measured cycle used about 150,000 to 250,000 input tokens (about $1); output, which
includes the model's thinking, is the main uncertainty. Expect **roughly $2 to $6 per cycle**. Cost grows with the number
of names the committee reviews: with the screen on that is 12 top-band names plus every
held name (about 25 in a 100-name backtest), so raise the budget accordingly.
`samples_per_analyst = 3` roughly triples the analyst part. An `--mode llm` backtest
prints its budget (the per-run cap times the number of cycles) and asks before starting.
The re-identification probe makes 400 calls; set `--budget`. The decision report shows
the actual cost. Identical calls are cached on disk (`$AIFIRM_CACHE_DIR`, else
`$XDG_CACHE_HOME/aifirm`, else `~/.cache/aifirm`), so repeating a backtest over the same
dates costs nothing.

To use Alpaca's **paper** API instead of the built-in simulator, set `broker = "alpaca"`
in `[execution]` and export `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`. The run lists
the orders and asks you to approve them (`--yes` approves without asking; use it only for
a supervised schedule). The adapter refuses live (real-money) endpoints unless the code is
changed. The shadow book always uses the simulator.

### 7. Track record and audit

```bash
aifirm score --config examples/firm.toml            # resolve due forecasts; Brier scores; paired ICs
aifirm verify-ledger --config examples/firm.toml    # check the hash chain against the last recorded head
aifirm verify-ledger --config examples/firm.toml --shadow
```

## The staged plan (S0 to S6)

The research report sets gates that must be passed in order before any result counts.
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) maps each one to commands,
report fields and thresholds, and marks what the code does not check for you.

| Stage | In brief |
|---|---|
| S0 | Validate the machinery without an LLM: tests, quant and offline backtests, ledger checks, results that survive 2× costs. |
| S1 | For each model, know the contaminated period: training cutoff plus a 31-day buffer (vendor dates; measuring the real cutoff is up to you). |
| S2 | Before any backtest on pre-cutoff data, pass the re-identification gate (`aifirm reid-probe`) with blind mode on. |
| S3 | On pre-cutoff data, tune only plumbing, risk and costs, never prompts. The trial ledger counts configurations; prompt edits are not counted unless you change the config's `name`. |
| S4 | Choose a design by paired per-date IC tests on several models' clean windows; require DSR ≥ 0.95, factor alpha t ≥ 2 and the 95th percentile against random books. Each backtest report computes these; the multi-model ladder is manual. |
| S5 | One untouched holdout on the production model's clean window. Not enforced. |
| S6 | Forward paper trading of one frozen configuration with the shadow book; `aifirm compare` gives a verdict after about two years of monthly decisions. Pilot capital and scaling are human decisions. |

## How to read the reports

**Decision report** (one per cycle):

- *Summary*: NAV, the final book's exposure, orders, warnings and LLM cost.
- *Screen* (if enabled): universe size, eligible names and the deep-dive set.
- *Committee*: per name, the final probability; in overlay mode also the quant core, the
  LLM pool and the tilt applied; the equal-weight pool (to see what track-record
  weighting changed); dispersion; whether a challenge ran and how many analysts changed
  sides; the chair's recommendation and multiplier; and the final weight. Tier-gate notes
  and calibration maps in force follow.
- *Views*: each analyst's sealed probability, revised probability, weight and grounding
  status. `FLAGGED` means the view cited an unknown fact, cited nothing, or used numbers
  the data desk never supplied; such a view counts half.
- *IC memos*, *Portfolio* (weights after code, after the risk review, after the risk
  engine, with every constraint that bound), *Risk engine*, *Orders*, *LLM usage*,
  *Warnings and data gaps*. "LLM OUTAGE" means half or more of the analyst calls (or of
  the required memos) failed, or the LLM budget ran out, so the book was held. "RISK
  REVIEW HALT" means no committee trades this cycle; the current book is kept.
- *Quant-only shadow book*: the shadow's decision and the comparison.

**Backtest report**: the six lines (window and assets; model and cutoff; cost model and
breakeven; configurations tried; deflated Sharpe; benchmark and random-book null);
metrics against the baselines; "Is there skill, and would we know?" (rank IC per signal,
paired IC tests, power, random books, cost sensitivity, factor attribution); the value the
language models added over the quant-only baseline net of their cost; the pre/post cutoff
split; exposure, turnover and costs; the forecast scorecard; and an *Honest
interpretation* generated from the numbers. Read the six lines and the interpretation
first.

## Data notes

- **Yahoo Finance** prices come through `yfinance`, which is not affiliated with Yahoo.
  Yahoo's terms allow personal use only. Before any business use, replace it with a
  licensed data vendor that includes delisted names, and store raw prices with corporate
  actions.
- **SEC EDGAR** asks automated clients to declare who they are and to stay under 10
  requests per second. Set `SEC_USER_AGENT` to your own name and contact email; the
  default is a placeholder. The code waits at least 0.15 seconds between SEC requests and
  caches responses on disk.
- **FRED** data is used under the Federal Reserve Bank of St. Louis terms of use. Some
  series are owned by third parties (for example the ICE BofA high-yield spread and the
  Cboe VIX) and carry their own copyright and redistribution limits; check each series'
  notes on FRED before you share data or results built on it.
- **Survivorship bias.** Yahoo does not serve most delisted tickers, and the bundled
  `sp100` list is today's membership. Names that were dropped, acquired or delisted are
  missing and later joiners are present, so backtests on it are biased upward. Reports
  say so. SEC's ticker-to-company map is current, not historical; pin a filer with
  `[universe.ciks]` when a ticker changed hands.
- **News** is live only (the free feed cannot be replayed point-in-time), so backtests
  never use headlines.

## Safety

- **Paper by default.** The default broker is a simulator. Alpaca orders need a human
  approval (unless you pass `--yes` or set `execution.require_approval = false`), and the
  adapter refuses live endpoints. Moving to real money, raising limits
  and unfreezing a book are human decisions.
- **Kill switch.** Create the file `<runs_dir>/KILL` (for `examples/firm.toml`:
  `examples/runs/KILL`) and no orders are generated, for the live, offline and shadow
  books, until you delete it.
- **Drawdown rules.** At a 10% drawdown from the peak the active positions are halved
  until half of the drawdown is recovered. At 20% the active positions go to cash (the
  passive core is kept) and the book stays frozen until a human resets it. After a one-day
  loss of 5% or more since the previous cycle, no position may grow that cycle.
- **Unfreeze.** Review what happened, then run

  ```bash
  aifirm unfreeze --config examples/firm.toml --reason "what you reviewed and why"   # add --offline or --shadow
  ```

  It shows the drawdown, asks for confirmation (`--yes` skips it), clears the freeze,
  restarts the high-water mark at the NAV recorded at the book's last decision, and writes
  the before and after state and your reason to the ledger.
- **Fail closed.** A failed LLM call is an abstention, never a default position. A failed
  memo on a name the book does not own gives it no position; on a held name it keeps the
  current weight, with a warning. A news-reader failure means no news this cycle. Data
  that cannot be loaded is shown to the agents as a gap. A broken ledger or state file
  stops the run.
- **Config mistakes stop the run.** An unknown key is an error, and fractions are
  range-checked: `max_position = 10` is refused with "use 0.10 for 10%".
- **Untrusted text.** Headlines are read only by a tool-less reader whose output code
  validates; source tiers are set by code; a bullish view resting only on unverified news
  is capped at 0.5.

## Not investment advice

This is research software. Its forecasts are probabilities, most of them close to a coin
flip, and simulated results do not predict future returns. Managing other people's money
brings legal obligations (investment-adviser registration, rules on advertising
performance, and more); see the research report and talk to a lawyer before going near
that.

## Project layout

```
aifirm/
  config.py          FirmConfig (TOML), validation, fingerprint, model training cutoffs
  schemas.py         LLM output schemas and internal records
  universes/         bundled universe lists (sp100.txt)
  data/              prices (Yahoo), fundamentals and filings (SEC), macro (FRED), news and source tiers, fact sheets
  llm/client.py      Claude client (structured outputs, cache, budget) and OfflineLLM
  agents/            roles, analysts, grounding, news reader, tier gate, committee, risk review, lessons,
                     re-identification probe, offline rules
  quant/             factor model (the "quant" analyst) and the broad-universe screen
  portfolio/         pooling, bounded overlay, portfolio construction, invariant risk engine
  execution/         paper broker with costs and dividends, guarded Alpaca adapter
  journal/           hash-chained ledger, forecast store and scoring, calibration and rank IC, lessons
  pipeline.py        gather_data and decide: one decision cycle, shared by live runs and backtests
  books.py           paper-book state and settlement (live and shadow books)
  shadow.py          the quant-only shadow book and the live-vs-shadow comparison
  backtest/          metrics, walk-forward engine, IC and power, random books, cost sensitivity, factors
  report.py          markdown decision and backtest reports
  cli.py             aifirm factsheet | run | compare | screen | backtest | score | reid-probe | verify-ledger | unfreeze
examples/firm.toml   the default configuration, commented
docs/                ARCHITECTURE.md, EVALUATION_PROTOCOL.md, research/RESEARCH_REPORT.md
tests/               offline tests on synthetic data
```

Configuration reference: [examples/firm.toml](examples/firm.toml) lists every setting with
its default and the reason for it.
