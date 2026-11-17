# aifirm: a research-grade multi-agent AI investment firm

`aifirm` runs a small investment firm made of software agents. A data desk builds verified,
point-in-time fact sheets. Several analyst agents each read a different slice of those facts
and write sealed, independent forecasts. The forecasts are pooled statistically, and analysts
argue only when they disagree. A committee chair audits the result and may only shrink it.
Deterministic code then sizes positions, enforces risk limits and sends orders to a paper
broker. Every step goes into a tamper-evident ledger, every forecast is scored when its
horizon passes, and those scores set each analyst's future weight.

```
Data desk (Yahoo prices, SEC filings as filed, FRED macro; every number has an ID)
   -> analysts: fundamental | technical | macro | events (Claude) + quant factor model (code)
   -> pool: believability-weighted trimmed log-odds mean (+ equal-weight shadow)
   -> challenge round only if analysts disagree -> private revisions
   -> IC memo by the chair (can only shrink the edge)
   -> portfolio engine (code): fractional Kelly, caps, volatility target, cash floor, turnover
   -> LLM risk review (can only shrink) -> risk engine (code; drawdown freeze, loss halt, kill switch)
   -> orders -> paper broker (fills at the next open, with costs) -> hash-chained ledger
   -> forecasts resolve -> Brier scores -> believability weights and lessons (point-in-time)
```

The design and the evidence behind it are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(principles P1 to P10) and [docs/research/RESEARCH_REPORT.md](docs/research/RESEARCH_REPORT.md).
In one line: **LLMs propose, code disposes.** Agents output probabilities and evidence. They
never choose weights, share counts or leverage.

## Honest expectations

- **There is no evidence that this, or any published LLM trading system, reliably beats the
  market.** Many published results disappear once fills happen at the next open, costs are
  charged, look-ahead leaks are removed and the number of variants tried is accounted for.
  This project is built to measure that honestly, not to promise returns.
- **Good one-month forecasts sit close to 0.50.** The analysts are told so, and the sizing
  rules trust small edges only a little (quarter Kelly, a 10% position cap, disagreement cuts
  size). With the defaults (`min_prob_to_hold = 0.52`, `core_weight = 0`) the book is often
  mostly cash. Set `portfolio.core_weight` (for example 0.5) if you want the rest of the money
  to track the benchmark instead.
- **Backtests before a model's training cutoff are contaminated.** The model may remember what
  happened next. The backtest report splits those periods out with a warning. Only decisions
  made after the cutoff (forward paper trading) can show skill.
- **The offline mode is not AI.** It replaces every LLM with simple deterministic rules, so you
  can run and test the whole machine without an API key. Never act on its output.

A first real-data run makes the point. A quant-only backtest (no LLM) on AAPL, MSFT, NVDA, JPM,
XOM and JNJ from 2023-01-01 to 2025-12-31 returned +15.0% (Sharpe 1.25, max drawdown -3.1%).
SPY returned +86.3% and an equal-weight basket of the six names +154.9%. The strategy was on
average only about 9% invested, and its forecasts had no skill over a coin flip (Brier skill
-0.010 over 210 forecasts). The report says all of this in its "Honest interpretation"
section.

## Quick start

Requires Python 3.10 or later.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC asks automated clients to identify themselves
```

`python -m aifirm ...` works the same as the `aifirm` command.

### 1. Offline demo (no API key)

```bash
aifirm factsheet AAPL                                   # the verified fact sheet the analysts read
aifirm run --config examples/firm.toml --offline        # one decision cycle, deterministic rules, paper broker
aifirm backtest --config examples/firm.toml --start 2023-01-01 --end 2025-12-31 --mode quant
aifirm backtest --config examples/firm.toml --start 2025-01-01 --end 2025-12-31 --mode offline
pytest                                                  # the whole test suite runs offline on synthetic data
```

The offline book lives in `runs/offline/`, separate from the real paper book, so heuristic
trades never mix with it.

### 2. Live paper trading with Claude

```bash
export ANTHROPIC_API_KEY=...        # or run `ant auth login` once to create a profile
aifirm run --config examples/firm.toml
```

Run it once per decision cycle (for example monthly, after the close). Each run:

1. fills the orders queued by the previous run at the **next trading day's open** after that
   decision, with commission, spread and market impact;
2. resolves forecasts whose horizon has passed and writes a few process lessons;
3. runs the committee for every ticker, sizes the book, applies the risk review and the risk
   engine, and queues new orders for the next open;
4. writes the report to `runs/reports/decision_<date>_<run_id>.md` and appends every step to
   `runs/ledger.jsonl`.

A run before the next open refuses to decide again, so a plan cannot be doubled. By default the
run decides on the last *completed* session (after 16:30 New York time), never on a partial day.

To use the Alpaca **paper** API instead of the built-in simulator, set `broker = "alpaca"` in
the config and export `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`. The run then lists the
orders and asks you to approve them (`--yes` approves without asking; use it only for a
supervised schedule). The adapter refuses live (real-money) endpoints unless code is changed
deliberately.

### 3. Backtests

```bash
aifirm backtest --config my.toml --start 2023-01-01 --end 2025-12-31 --mode quant     # no LLM
aifirm backtest --config my.toml --start 2023-01-01 --end 2025-12-31 --mode offline   # plumbing check
aifirm backtest --config my.toml --start 2026-07-01 --end 2026-11-14 --mode llm       # Claude; costs money
```

Every backtest decides at the close and fills at the next open with costs, marks to market
daily, resolves forecasts as time passes and uses only outcomes known by each decision date.
It runs three baselines through the same engine: benchmark buy-and-hold, an equal-weight
universe with the same rebalance dates and costs, and (except in quant mode) the quant
analyst alone. Each run is appended to `runs/trials.jsonl`; the number of distinct
configurations you have tried feeds the deflated Sharpe ratio, so keep that file. The report
is written to `runs/backtests/<run_id>/report.md` next to that run's ledger and forecasts.

### 4. Track record and audit

```bash
aifirm score --config examples/firm.toml            # resolve due forecasts, print Brier scores per agent
aifirm verify-ledger --config examples/firm.toml    # check the hash chain against the last recorded head
```

## How to read the reports

**Decision report** (one per cycle):

- *Committee*: per ticker, the pooled probability that the stock beats the benchmark over the
  horizon, the same pool with equal weights (to see what track-record weighting changed),
  dispersion (analyst disagreement), whether a challenge round ran and how many analysts
  changed sides, the chair's recommendation and edge multiplier, and the final weight.
- *Views*: each analyst's sealed probability, revised probability, weight and grounding status.
  `FLAGGED` means the view cited a fact ID that is not on its sheet or used several numbers the
  data desk never supplied; such a view counts half.
- *IC memos*: thesis, variant perception, base rate, pre-mortem, kill criteria and checklist.
- *Portfolio*: the weights proposed by code, after the LLM risk review and after the risk
  engine, with every constraint that bound.
- *Risk engine*, *Orders*, *LLM usage* (calls, tokens, dollars), *Warnings and data gaps*.
  "LLM OUTAGE" means half or more of the model calls failed, so the book was held unchanged.

**Backtest report**: metrics against the baselines; probabilistic and deflated Sharpe ratios
and the minimum track record; the pre/post training-cutoff split; turnover and costs; the
forecast scorecard with a calibration table; and an *Honest interpretation* section generated
from the numbers. Read that section first.

## Cost estimate per cycle (Claude)

Measured on a real cycle: an analyst prompt is about 2,800 input tokens, an IC memo prompt
about 5,900, a challenge about 4,200, a private revision about 2,200 and the risk review about
3,000 plus roughly 1,000 per held position. With the default 10 tickers and
`claude-opus-5-5` ($4 per million input tokens, $20 per million output tokens; check current
prices) a cycle makes about 40 analyst calls, memos for the names with an edge or a position,
occasional challenge rounds, one risk review and up to 8 lessons. That comes to about 150,000
to 250,000 input tokens (about $1) plus the output, which includes the model's thinking and is
the main uncertainty: **roughly $2 to $6 per cycle** in total. `samples_per_analyst = 3`
roughly triples the analyst part. Each run stops at `process.max_llm_cost_usd_per_run`
(default $10); an exhausted budget holds the book rather than trading on partial views. The
report shows the actual cost. Identical calls are cached on disk, so repeating a backtest over
the same dates costs nothing.

## Safety and legal notes

- **Paper first.** The default broker is a simulator. Moving to real money, raising limits and
  unfreezing a book after a drawdown freeze are human decisions. To unfreeze, review what
  happened, then set `"frozen": false` in `runs/risk_state.json` yourself.
- **Kill switch:** create the file `runs/KILL` and no orders are generated until you delete it.
- **Fail closed:** a failed LLM call is an abstention, never a default position; a required
  memo that fails sets that name's edge to 0; data that cannot be loaded is shown to the
  agents as a gap; a broken ledger stops the run.
- **Data limits:** Yahoo does not serve most delisted tickers, so a universe chosen today is
  biased toward survivors. SEC's ticker-to-company map is current, not historical: after a
  corporate reorganization (for example a new holding company), older fundamentals for that
  ticker can be missing, and the report lists such gaps. Live headlines are never used in
  backtests because they cannot be replayed point-in-time.
- **Not investment advice.** This is research software. Managing other people's money brings
  legal obligations (investment-adviser registration, rules on advertising performance, and
  more); see the research report and talk to a lawyer before going near that.

## Project layout

```
aifirm/
  config.py          FirmConfig (TOML) and model training cutoffs
  schemas.py         LLM output schemas and internal records
  data/              prices (Yahoo), fundamentals and filings (SEC), macro (FRED), news, fact sheets
  llm/client.py      Claude client (structured outputs, cache, budget) and OfflineLLM
  agents/            analyst roles and prompts, grounding check, committee, risk review, lessons, offline rules
  quant/signals.py   cross-sectional factor model (the deterministic "quant" analyst)
  portfolio/         pooling, portfolio construction, invariant risk engine
  execution/         paper broker with costs, guarded Alpaca adapter
  journal/           hash-chained ledger, forecast store and scoring, point-in-time lessons
  pipeline.py        gather_data and decide: one decision cycle, shared by live runs and backtests
  backtest/          metrics (Sharpe, PSR, DSR, MinTRL) and the walk-forward engine
  report.py          markdown decision and backtest reports
  cli.py             aifirm factsheet | run | backtest | score | verify-ledger
examples/firm.toml   the default configuration, commented
docs/                ARCHITECTURE.md (design and module contracts), research/RESEARCH_REPORT.md
tests/               offline tests on synthetic data
```

Configuration reference: [examples/firm.toml](examples/firm.toml) lists every setting with its
default and the reason for it.
