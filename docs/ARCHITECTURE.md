# Architecture of the AI investment firm (`aifirm`)

This document is both the design rationale and the **module contract** that every
component is built against. The research behind each choice is in
[`docs/research/RESEARCH_REPORT.md`](research/RESEARCH_REPORT.md).

## 1. The one-paragraph version

The firm runs a **decision cycle** (daily, weekly or monthly). First, a deterministic **Data Desk**
builds a point-in-time **fact sheet** for every ticker. Every number gets an ID and a source.
Next, several **analyst agents** each read a *different slice* of the facts and write
**sealed, independent forecasts**: P(stock beats benchmark over the horizon), expected excess return,
bull/base/bear scenarios, and evidence that cites fact IDs. One analyst is a deterministic quant
factor model. A **grounding checker** flags any number that is not in the fact sheet.
The forecasts are **pooled statistically**: a believability-weighted, trimmed mean in log-odds
space, not an LLM consensus. Only when the analysts **disagree** does a **challenge round** run:
the most opposed analyst writes the strongest counter-case, then everyone re-estimates privately.
The **committee chair** writes a machine-checkable **IC memo** (thesis, base rate, pre-mortem,
kill criteria, checklist) and may only *shrink* the edge. A **deterministic portfolio engine**
converts edges into weights (probability-implied edge, fractional Kelly, volatility target,
position/sector caps, cash floor, core-satellite, no-trade band, turnover cap). An LLM
**risk reviewer** may only reduce positions. A deterministic **risk engine** (the "invariant
core") then enforces drawdown de-risking, a daily-loss halt, restricted lists, liquidity caps
and a kill switch. Orders go to a **paper broker** (or Alpaca, with human approval). Everything is
written to a **hash-chained ledger**. Every forecast is **scored** when its horizon passes, and
those scores set each analyst's future weight. **Lessons** are retrieved only if they were known
by the decision date.

```
             ┌──────────── Data Desk (deterministic, point-in-time) ────────────┐
 prices ───► │ price/trend/risk facts │ SEC XBRL as-filed │ FRED macro │ 8-K events│──► FactSheet per ticker
             └──────────────────────────────────────────────────────────────────┘        (IDs + sources)
                    │ different slices                                    
     ┌──────────────┼───────────────┬──────────────┬───────────────┐
     ▼              ▼               ▼              ▼               ▼
 Fundamental    Technical        Macro/regime    Events/news     Quant factor
 (LLM)          (LLM)            (LLM)           (LLM)           (code)
     └─── sealed AnalystViews (prob, E[excess], scenarios, evidence→fact IDs) ───┐
                         grounding check  ─────────────────────────────────────►│
                                                                                ▼
                  Pool: believability-weighted trimmed log-odds mean (+ equal-weight shadow)
                                       │ dispersion > threshold?
                                       ├── yes ─► Challenge (most-opposed analyst) ─► private Revisions ─► re-pool
                                       ▼
                       IC memo by chair (thesis, base rate, pre-mortem, kill criteria; can only shrink)
                                       ▼
            Portfolio engine (code): edge → fractional Kelly → caps → vol target → core/cash → band/turnover
                                       ▼
            LLM risk review (can only scale down)  ─►  Risk engine (code, cannot be overridden)
                                       ▼
                   Orders ─► human approval (live) ─► Broker (paper / Alpaca)  ─► Ledger
                                       ▼
            Forecast store ─► resolve at horizon ─► Brier/calibration ─► believability weights, lessons
```

## 2. Design principles (and the evidence)

| # | Principle | Why (see research report) |
|---|-----------|---------------------------|
| P1 | **LLMs propose, code disposes.** Agents output probabilities and evidence. They never output share counts, weights or leverage. | FinCon's CVaR ablation; Alpha Arena season 1 losses from leverage/overtrading; ai-hedge-fund v2 "conviction requests, risk disposes" |
| P2 | **Every number comes from the Data Desk.** Agents cite fact IDs; a grounding check flags unsupported numbers. | FinanceBench error rates; TradingAgents hallucination fixes; FinRobot's numeric-grounding auditors |
| P3 | **Point-in-time everything.** Fundamentals as filed, macro with publication lags, no live-only feeds in backtests, decide at close T and execute at the open of T+1. | Look-ahead bugs fixed repeatedly in TradingAgents; FINSABER; the "Profit Mirage" and Look-Ahead-Bench studies |
| P4 | **Independent first, aggregate statistically, debate only on disagreement.** | "Debate or Vote" (NeurIPS 2025); Lorenz 2011; Tetlock/Good Judgment aggregation |
| P5 | **Information diversity, not persona diversity.** Analysts get different data slices. Optional model diversity. No famous-investor personas. | Hidden-profile research (Stasser & Titus); persona prompting shows no accuracy gain |
| P6 | **Authentic dissent.** The challenger is the analyst whose own independent view most opposed the pool. | Nemeth 2001 (authentic dissent beats assigned devil's advocates) |
| P7 | **Score everything; weight by track record.** Each view is a resolvable forecast with a Brier score. Weights are shrunk toward equal, with a minimum sample, floor and cap. The equal-weight result is logged alongside. | Superforecasting; Bridgewater believability; ContestTrade ablation |
| P8 | **Separate idea generation from sizing.** Sizing is deterministic (fractional Kelly, volatility targeting, caps). | Best-ideas literature; sizing skill studies; full Kelly is ruinous with estimated edges |
| P9 | **An invariant risk core.** Drawdown de-risk/freeze (pod-style), daily-loss halt, liquidity caps, restricted list, kill switch. Nothing an LLM says can override it. | Multi-manager pod practice; SEC 15c3-5 / Knight Capital |
| P10 | **Honest evaluation.** Next-open fills, costs and impact, baselines (benchmark, equal weight, quant-only), deflated Sharpe with a trial ledger, pre/post model-cutoff split, forward paper trading before real money. | FINSABER; Bailey & López de Prado; StockBench/DeepFund live results |

## 3. Package layout and module contracts

```
aifirm/
  config.py            FirmConfig (TOML-loadable), MODEL_TRAINING_CUTOFFS                [done]
  schemas.py           LLM output schemas + internal records + sanitize_view               [done]
  data/                cache, http, prices, fundamentals (SEC), macro (FRED), news, factsheet [done]
  llm/client.py        LLM interface, ClaudeLLM, OfflineLLM, Usage/budget, CallRecord      [done]
  agents/
    roles.py           AnalystRole specs (task, checklist, data slice)
    analysts.py        prompts, run_analyst, blind mode
    grounding.py       numeric grounding checker
    committee.py       run_committee: views → pool → challenge → revisions → memo
    risk_review.py     LLM portfolio risk review (can only scale down)
    reflection.py      lessons from resolved forecasts
    offline.py         make_offline_llm(): deterministic responders for every role
  quant/signals.py     cross-sectional factor model → deterministic "quant" analyst view
  portfolio/
    aggregate.py       pool_probabilities, aggregate_views
    construct.py       construct_portfolio
    risk_engine.py     RiskState, apply_risk_limits, orders_from_target
  execution/broker.py  PaperBroker (cost model), AlpacaBroker (paper/live REST)
  journal/
    ledger.py          append-only hash-chained JSONL ledger
    forecasts.py       ForecastStore (SQLite), resolve, scorecard, believability_weights
    lessons.py         LessonStore (point-in-time retrieval)
  pipeline.py          gather_data, decide (one decision cycle, shared by live and backtest)
  backtest/
    metrics.py         Sharpe, Sortino, MDD, PSR, DSR, MinTRL, Brier                       [done]
    engine.py          walk-forward backtest with baselines and cutoff split
  report.py            markdown decision and backtest reports
  cli.py               `aifirm factsheet | run | backtest | score | verify-ledger`
```

### 3.1 Existing contracts (already implemented; read the code)

* `aifirm.data.factsheet.FactSheet`/`Fact`: `fs.facts` (list of `Fact(id,label,value,unit,as_of,source)`),
  `fs.get(id)`, `fs.ids()`, `fs.render(sections=(...))` where sections ⊂ `("price","fund","macro","events","news")`,
  `fs.events`, `fs.headlines`, `fs.data_gaps`, `fs.ticker`, `fs.as_of`, `fs.name`, `fs.sector`.
  Fact IDs: `TICKER.key` for ticker facts (e.g. `NVDA.ret_3m`, `NVDA.pe_ttm`, `NVDA.vol_60d`,
  `NVDA.mom_12_1`, `NVDA.roe`, `NVDA.earnings_yield`, `NVDA.fcf_yield`, `NVDA.operating_income_margin`, `NVDA.adv_20d`, `NVDA.price`),
  `MKT.SPY_*` and `MACRO.<FRED id>` for shared facts. `aifirm.data.factsheet.PRICE_KEYS` lists the price-derived keys.
* `aifirm.data.prices.PriceData` (`close`, `adj_close`, `volume`, `open` DataFrames indexed by date, columns = tickers;
  `.as_of(date)`, `.returns()`), `load_prices(tickers, start, end)`, `from_frames(close, adj_close, volume, open_)`.
* `aifirm.data.fundamentals.load_fundamentals(ticker, as_of)`, `recent_filings(ticker, as_of, lookback_days)`;
  `aifirm.data.macro.macro_snapshot(as_of)`; `aifirm.data.news.recent_headlines(ticker, as_of)` (live only);
  `aifirm.data.factsheet.build_factsheet(ticker, as_of, prices, fundamentals, macro, filings, headlines, sector, benchmark)`.
* `aifirm.llm.client.LLM.structured(role=, model=, system=, user=, schema=, effort=, max_tokens=) -> (obj, CallRecord)`;
  raises `LLMCallFailed` (caller must turn it into an abstention / no-trade, never a default position) and `BudgetExceeded`.
  `OfflineLLM(responders)` dispatches on the longest matching **role prefix**. Role strings used across the system:
  `analyst.<name>`, `challenge`, `revision.<name>`, `memo`, `risk_review`, `lesson`.
* `aifirm.schemas`: `AnalystView`, `Scenario`, `EvidenceItem`, `Challenge`, `Revision`, `ICMemo`, `KillCriterion`,
  `ChecklistItem`, `RiskReview`, `RiskConcern`, `Lesson`, `ViewRecord`, `AggregatedView`, `TargetPosition`,
  `TargetPortfolio`, `Order`, `Fill`, `ForecastRecord`, `clamp`, `sanitize_view`.
* `aifirm.config.FirmConfig` and sub-configs; `model_cutoff(model)`.
* `aifirm.backtest.metrics.performance_report(returns, benchmark_returns, turnover, costs, n_trials)`, `brier_score`.

### 3.2 `agents/roles.py`

```python
@dataclass(frozen=True)
class AnalystRole:
    name: str                    # "fundamental" | "technical" | "macro" | "events"
    title: str
    mandate: str                 # what this analyst is responsible for judging
    checklist: tuple[str, ...]   # concrete things to evaluate (task-specific, not persona)
    sections: tuple[str, ...]    # FactSheet.render sections this analyst sees (its data slice)
    extra_fact_keys: tuple[str, ...] = ()  # specific facts from other sections it may also see (e.g. price, market_cap)
ROLES: dict[str, AnalystRole]
```
Data slices: fundamental → `fund` (+ `price`, `ret_12m` facts); technical → `price`; macro → `macro` + the ticker's
`beta_1y`, `vol_60d`, `ret_3m`, `rel_ret_3m` and sector; events → `events`, `news` + `ret_1m`, `price`, `vol_20d`.

### 3.3 `agents/analysts.py`

```python
def analyst_system_prompt(role: AnalystRole, horizon_days: int, benchmark: str) -> str   # stable text → prompt-cacheable
def analyst_user_prompt(role: AnalystRole, fs: FactSheet, horizon_days: int, benchmark: str,
                        lessons: list[str], blind: bool) -> str
def run_analyst(llm: LLM, role_name: str, fs: FactSheet, cfg: FirmConfig, model: str, sample: int = 0,
                lessons: list[str] | None = None) -> ViewRecord
def blind_factsheet(fs: FactSheet) -> FactSheet   # alias ticker → "STOCK", hide name, drop absolute-USD facts, relative dates
```
Prompt requirements: neutral framing (no client thesis, no third-party ratings); calibration guidance (for a ~1-month
horizon, honest probabilities usually sit between 0.35 and 0.65 and the base rate is about 0.5);
"cite fact IDs, never introduce a number not on the fact sheet"; "abstain if your slice is insufficient";
explain `private_insight`; ask for consistency between `prob_outperform`, scenarios and expected excess return.
`run_analyst` catches `LLMCallFailed` → returns an abstaining `ViewRecord` (with `grounding_issues=["llm failure: ..."]`),
applies `sanitize_view`, runs the grounding check, sets `grounding_ok` and `grounding_issues`.

### 3.4 `agents/grounding.py`

```python
@dataclass
class GroundingReport:
    ok: bool
    unknown_fact_ids: list[str]
    unsupported_numbers: list[str]
    checked_numbers: int
def check_view(view: AnalystView, fs: FactSheet, rel_tol: float = 0.03) -> GroundingReport
```
Extract numbers from free-text fields (evidence points, scenario descriptions, private insight, risks). Ignore
years (1900-2100), small counting numbers ≤ 12 that are followed by time units (e.g. "3 months", "12-month"), and
the view's own probabilities/returns. A number is supported if it matches some fact value under display variants
(raw, ×100 for ratios shown as %, ÷1e3/1e6/1e9/1e12 for USD) within `rel_tol` or 0.05 absolute after rounding.
`ok` is false if any cited fact ID is unknown or more than 2 numbers are unsupported.

### 3.5 `agents/committee.py`

```python
@dataclass
class CommitteeResult:
    ticker: str
    views: list[ViewRecord]
    aggregate: AggregatedView
    challenge: Challenge | None
    challenger: str | None
    memo: ICMemo | None
    edge_multiplier: float          # memo.conviction_multiplier clamped to [0,1]; 1.0 if no memo; 0 if memo says review/avoid
def run_committee(llm: LLM, fs: FactSheet, cfg: FirmConfig, *, quant_view: ViewRecord | None,
                  believability: dict[str, float], lessons: dict[str, list[str]], held: bool,
                  log: Callable[[str, dict], None] = lambda e, p: None) -> CommitteeResult
```
1. For each enabled LLM analyst and each sample, call `run_analyst` (analyst models rotate across samples; run calls
   concurrently with a thread pool of `cfg.models.max_parallel_calls`). Add `quant_view` if given.
2. Set each view's `weight` = `believability.get(agent, 1.0)`. Ungrounded views get their weight halved. Abstentions are excluded from pooling.
3. Pool with `portfolio.aggregate.aggregate_views`.
4. If `challenge_enabled` and `dispersion > challenge_dispersion_threshold` and ≥ 2 non-abstaining views: the challenger is the
   analyst whose prob is farthest from the pooled prob *on the opposite side of 0.5* (if none, the farthest overall). It writes
   a `Challenge` (role `challenge`) seeing the full fact sheet plus an anonymized summary of the pooled view (no agent names).
   Each other LLM analyst writes a private `Revision` (role `revision.<name>`) seeing only its own slice, its own sealed view and the
   challenge. Store `revised_prob`/`revised_excess_pct` and re-pool with revised values. Count `flips` (crossing 0.5).
5. If `memo_enabled` and (`|pooled − 0.5| ≥ memo_min_edge` or `held`): the chair (role `memo`, `committee_model`,
   `committee_effort`) writes an `ICMemo` from the full fact sheet + all views (order randomized deterministically by ticker hash,
   agent names replaced by letters) + the challenge. `edge_multiplier` = clamp(conviction_multiplier, 0, 1), forced to 0 when the
   recommendation is `review` or `avoid`/`sell` for a long-only book.
6. Call `log(event_type, payload)` for views, aggregate, challenge, revisions and memo, so the pipeline can write them to the ledger.

### 3.6 `agents/risk_review.py`, `agents/reflection.py`, `agents/offline.py`

```python
def review_portfolio(llm: LLM, target: TargetPortfolio, factsheets: dict[str, FactSheet], cfg: FirmConfig) -> RiskReview | None
def apply_risk_review(target: TargetPortfolio, review: RiskReview | None) -> TargetPortfolio   # multipliers clamped to [0,1]; halt → all active weights 0
def write_lesson(llm: LLM, forecast: ForecastRecord, view_summary: str, cfg: FirmConfig) -> Lesson | None
def make_offline_llm() -> OfflineLLM
```
The offline responders parse the rendered fact lines (`[ID] label: value (...)`) from the prompt and apply simple, transparent
rules (e.g. technical: momentum and trend; fundamental: earnings and FCF yield, growth, margins; macro: beta × regime;
events: count of negative 8-K items). They return valid schema objects whose evidence cites real fact IDs from the prompt.

### 3.7 `quant/signals.py`

```python
def factor_scores(factsheets: list[FactSheet]) -> dict[str, dict[str, float]]   # per ticker: z-scores per factor + composite
def quant_views(factsheets: list[FactSheet], horizon_days: int) -> dict[str, ViewRecord]  # agent="quant", model="deterministic"
```
Factors (cross-sectional z-scores, winsorized at ±3, missing → 0): momentum (`mom_12_1`), low volatility (−`vol_60d`),
quality (`roe`, `operating_income_margin`), value (`earnings_yield`, `fcf_yield`). Composite = mean of the available factor z's.
Probability mapping is deliberately modest: `p = 0.5 + 0.06·tanh(composite)`. Expected excess = σ_h·Φ⁻¹(p), with σ_h from `vol_60d`.
Evidence cites the fact IDs used.

### 3.8 `portfolio/aggregate.py`

```python
def pool_probabilities(probs: list[float], weights: list[float] | None = None, trim: float = 0.2) -> float
def aggregate_views(ticker: str, views: list[ViewRecord], trim: float = 0.2, use_revised: bool = False) -> AggregatedView
```
Weighted trimmed mean in log-odds space (sort by logit, drop `trim` of total weight from each tail when n ≥ 4, clip probs to
[0.01, 0.99]). Expected excess = weighted median of member expected excess returns. Dispersion = weighted std-dev of
member probabilities. `prob_equal_weight` = the same pooling with unit weights. With no non-abstaining views: prob 0.5, excess 0.

### 3.9 `portfolio/construct.py`

```python
def construct_portfolio(aggs: dict[str, AggregatedView], edge_multipliers: dict[str, float],
                        factsheets: dict[str, FactSheet], current_weights: dict[str, float],
                        cfg: FirmConfig, as_of: str, sectors: dict[str, str],
                        returns: pd.DataFrame | None = None) -> TargetPortfolio
```
Per ticker (horizon h = `horizon_days`, σ_ann from `vol_60d` fact, fallback 30%; σ_h = σ_ann·√(h/252)):
`μ_prob = σ_h·Φ⁻¹(p)`, `μ_exp = expected_excess/100`, `μ = 0.5·(μ_prob + μ_exp)`, then
`μ *= edge_multiplier · max(0, 1 − dispersion_shrink·dispersion)`. Long-only: weight 0 unless `p ≥ min_prob_to_hold` and μ > 0.
Raw weight = `kelly_fraction · μ / σ_h²`, capped at `max_position` (and at −`max_position` if shorting is allowed).
Then: sector caps (scale names in a breaching sector proportionally), sleeve volatility targeting using the sample covariance of
the last 126 days of `returns` if provided (else diagonal with 0.3 average correlation): if the sleeve's vol > `target_vol`, scale it down.
Then gross ≤ `max_gross` and cash ≥ `min_cash`. Apply core-satellite: final active weights × (1 − `core_weight`); `core_weight`
goes to the benchmark. No-trade band: keep the current weight when |target − current| < `no_trade_band`. Turnover cap:
if Σ|Δw| > `max_turnover`, scale all Δw proportionally. Record every constraint that binds in `constraints_hit` / `notes`.
Tickers that are currently held but absent from `aggs` are sold (weight 0) unless the turnover cap prevents it.

### 3.10 `portfolio/risk_engine.py` (invariant core)

```python
@dataclass
class RiskState:
    peak_nav: float
    frozen: bool = False
    last_nav: float | None = None
def update_risk_state(state: RiskState, nav: float) -> RiskState
def apply_risk_limits(target: TargetPortfolio, state: RiskState, nav: float, prev_nav: float | None,
                      current_weights: dict[str, float], cfg: FirmConfig, factsheets: dict[str, FactSheet],
                      runs_dir: str) -> tuple[TargetPortfolio, list[str]]
def orders_from_target(target: TargetPortfolio, positions: dict[str, float], prices: dict[str, float], nav: float,
                       adv: dict[str, float], cfg: FirmConfig, benchmark: str) -> tuple[list[Order], list[str]]
```
Rules: kill-switch file `runs_dir/<kill_switch_file>` → target = current weights (no trades) and message. Drawdown from peak
≥ `drawdown_freeze` → state.frozen=True and active weights → 0 (core stays). ≥ `drawdown_derisk` → active weights × 0.5.
One-day loss ≥ `daily_loss_halt` → no weight may increase this cycle. Restricted tickers and price < `min_price` → weight 0.
Orders: quantity = (target_w·nav − current value)/price, skipping |Δnotional| < `no_trade_band`·nav; cap each order's notional at
`max_adv_participation`·ADV (message when capped); sells before buys. The engine never raises gross exposure above the target.

### 3.11 `execution/broker.py`

```python
class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...       # shares
    def cash(self) -> float: ...
    def nav(self, prices: dict[str, float]) -> float: ...
    def submit(self, orders: list[Order], fill_prices: dict[str, float], date: str,
               adv: dict[str, float] | None = None, daily_vol: dict[str, float] | None = None) -> list[Fill]: ...
class PaperBroker:  # __init__(cash, cfg: CostConfig, state_path: str | None = None); save()/load()
class AlpacaBroker: # env ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY / ALPACA_BASE_URL (default paper); market orders
def execution_cost(notional: float, price: float, adv: float | None, daily_vol: float | None, cfg: CostConfig) -> tuple[float, float]
    # returns (commission, slippage_cost); slippage bps = slippage_bps + impact_coefficient·1e4·daily_vol·sqrt(|notional|/adv)
```
Paper fills at `fill_prices` (the next open in backtests), buys pay slippage up / sells receive down. Cash cannot go negative:
buys are scaled down to available cash. Fractional shares allowed.

### 3.12 `journal/`

```python
class Ledger:            # ledger.py
    def __init__(self, path: str): ...
    def append(self, event: str, payload: dict) -> str     # returns entry hash; entry = {seq, ts, event, payload, prev_hash, hash}
    def verify(self) -> tuple[bool, int]                  # (ok, entries checked)
class ForecastStore:     # forecasts.py (sqlite3)
    def __init__(self, path: str): ...
    def add(self, records: list[ForecastRecord]) -> None
    def all(self) -> list[ForecastRecord]
    def resolve(self, prices: PriceData, as_of: str) -> list[ForecastRecord]   # resolve where resolve_on <= as_of and data exists
def make_forecast_records(run_id, as_of, views: list[ViewRecord], agg: AggregatedView, cfg, trading_calendar) -> list[ForecastRecord]
def scorecard(records: list[ForecastRecord], as_of: str | None = None) -> dict[str, dict]   # per agent: n, brier, brier_skill, log_loss, hit_rate, calibration buckets
def believability_weights(records, as_of, agents, min_n=20, prior_n=30, floor=0.25, cap=2.0) -> dict[str, float]
class LessonStore:       # lessons.py (JSONL)
    def add(self, lesson: Lesson, resolved_on: str) -> None
    def retrieve(self, agent: str, as_of: str, k: int = 5) -> list[str]           # only lessons resolved on/before as_of
```
Resolution: outcome = 1 if the ticker's adjusted-close return from `as_of` to `resolve_on` beats the benchmark's. `resolve_on` is the
date `horizon_days` trading days after `as_of`. Only records resolved on or before the query date count toward weights (point-in-time).
Believability: skill = (0.25 − brier) / 0.25 per agent; weight = 1 + shrink·k·skill with shrink = n/(n+prior_n), k = 4, clipped to
[floor, cap]; weight = 1.0 when n < min_n. Forecasts flagged `pre_cutoff` are excluded from weights by default.

### 3.13 `pipeline.py`, `backtest/engine.py`, `report.py`, `cli.py`

```python
def gather_data(cfg, as_of, live: bool, prices: PriceData | None = None) -> tuple[PriceData, dict[str, FactSheet]]
def decide(cfg, llm, as_of, prices, factsheets, *, current_weights, nav, prev_nav, risk_state,
           forecast_store, lesson_store, ledger, run_id) -> Decision
def run_backtest(cfg, start, end, mode: "llm"|"offline"|"quant", llm=None, prices=None, fundamentals_loader=None, macro_loader=None) -> BacktestResult
```
The backtest decides at the close of each rebalance date and fills at the next trading day's **open** with costs. It marks to
market daily, resolves forecasts as time passes, and computes believability weights and lessons only from information
resolved by each decision date. Baselines run through the same engine: benchmark buy-and-hold, equal-weight universe
(same rebalance schedule and costs), and quant-only. Results are split into pre-cutoff (contaminated) and post-cutoff periods
for the analyst model. Each run appends to `runs/trials.jsonl`, and the number of distinct config fingerprints feeds the deflated Sharpe.

## 4. What is deliberately *not* automated

* Moving from paper to real money, raising capital limits, and resetting a frozen book are **human decisions**.
* Managing other people's money brings legal obligations (adviser registration, marketing rules for performance
  claims). See the research report section on the staged path. This software is a research tool, not investment advice.
