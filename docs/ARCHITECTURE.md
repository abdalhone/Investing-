# Architecture of the AI investment firm (`aifirm`)

This document is the design rationale and the **module contract** the code is built
against. It describes the code as it is, including where it falls short of the research.
The evidence behind each choice is in
[`docs/research/RESEARCH_REPORT.md`](research/RESEARCH_REPORT.md) (cited below as
"RESEARCH" plus a section number). How to judge whether the firm has any skill is in
[`docs/EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

## 1. The one-paragraph version

The firm runs a **decision cycle** at the close of a completed trading session (live:
whenever you run it, typically monthly; in backtests every 21 trading days by default).
Forecasts are about the next 21 trading days. A deterministic **Data Desk** builds a
point-in-time **fact sheet** for every ticker in the universe: every number gets an ID and
a source. A deterministic **quant factor model** scores every name. When the **broad
screen** is on, it ranks a wide universe and only the top band plus every held name go on
to an LLM **deep dive**. Live headlines of the names under review (the deep-dive set, or
the whole universe without the screen) are read by a **quarantined news reader**, a
tool-less model call whose output is typed events with a **source tier** set by code; no
other agent ever sees headline text. Four **LLM analysts** each read a
*different slice* of the facts and write **sealed, independent forecasts**: the
probability that the stock beats the benchmark over the horizon, an expected excess
return, scenarios, and evidence that cites fact IDs. A **grounding checker** flags numbers
that are not on the sheet. Before pooling, code caps any bullish view that rests only on
unverified news (the **tier gate**) and applies each analyst's **calibration map**, fitted
only on its own resolved forecasts. The LLM views are **pooled statistically** (a
believability-weighted, trimmed mean in log-odds space). By default that pool may only
**tilt the quant's probability by at most 0.05** (the **bounded overlay**). A
**challenge round** runs only when the members disagree. A **committee chair** writes a
machine-checkable **IC memo** and may only *shrink* the edge. A **deterministic portfolio
engine** turns edges into weights. An LLM **risk reviewer** may only reduce positions. A
deterministic **risk engine** then enforces drawdown rules, a daily-loss halt, caps and a
kill switch. Orders go to a **paper broker** (or Alpaca's paper API with human approval)
and fill at the **next open**. Everything is written to a **hash-chained ledger**. Every
forecast is **scored** when its horizon passes. Beside the live book, a **quant-only
shadow book** decides at the same close with every LLM stage removed, so the value the
language models add, net of what they cost, is measured forward.

```
            ┌───────────── Data Desk (code, point-in-time) ──────────────┐
 prices ──► │ price/trend/risk │ SEC XBRL as filed │ FRED macro │ filings │──► FactSheet per ticker (IDs + sources)
            └─────────────────────────────────────────────────────────────┘
                     │
   quant factor model scores the whole universe ──► screen (optional): top N by quant P > 0.5, plus held names
                     │                                                         │ deep-dive set only
   raw headlines ──► quarantined news reader (LLM, no tools) ──► typed events, tier A/B/C set by code ──► news facts
                     │
     ┌───────────────┼───────────────┬───────────────┬───────────────┐
     ▼               ▼               ▼               ▼               ▼
 Fundamental     Technical       Macro/regime    Events/news     Quant factor model
 (LLM)           (LLM)           (LLM)           (LLM)           (code)
     └── sealed views (P, E[excess], scenarios, evidence → fact IDs) + grounding check ──┐
                                                                                         ▼
       LLM views: tier gate (cap unsupported bullish views at 0.5) → calibration map per analyst
                                                                                         ▼
       LLM pool: believability-weighted trimmed log-odds mean (+ equal-weight pool logged)
       overlay: P_final = P_quant + clip(P_llm − 0.5, ±max_tilt)          (or "pool": quant is one member)
                    │ member dispersion > threshold? ── yes ──► challenge ──► private revisions ──► re-pool
                    ▼
       IC memo by the chair (can only shrink the edge)
                    ▼
       Portfolio engine (code): edge → quarter Kelly → 8% cap → sectors → vol target → band, half-way steps, turnover
                    ▼
       LLM risk review (can only scale down) ──► Risk engine (code; cannot be overridden)
                    ▼
       Orders ──► human approval (Alpaca) ──► paper broker, fills at the next open ──► ledger
                    ▼
       Forecast store ──► resolve at horizon ──► Brier, rank IC ──► weights, calibration, overlay bound, lessons
                    │
       Quant-only shadow book: same close, same sheets, same limits, no LLM ──► `aifirm compare`
```

## 2. Design principles (and the evidence)

| # | Principle | Why (see RESEARCH) |
|---|-----------|--------------------|
| P1 | **LLMs propose, code disposes.** Agents output probabilities and evidence. They never output share counts, weights or leverage. | FinCon's CVaR ablation; Alpha Arena season 1 losses from leverage and overtrading; ai-hedge-fund v2 "conviction requests, risk disposes" (1, 2.2) |
| P2 | **Every number comes from the Data Desk, and outside text is data.** Agents cite fact IDs; a grounding check flags unsupported numbers. Raw headlines reach only a quarantined, tool-less reader; its typed output is validated and tiered by code. | FinanceBench error rates; TradingAgents hallucination fixes; FinRobot's numeric auditors; CaMeL-style quarantined readers, TradeTrap and FARSIGHT (4, 5.5, 9.3 item 3) |
| P3 | **Point-in-time everything.** Fundamentals as filed and accepted before the close, macro with publication lags, headlines published before the session's New York close (16:00, 13:00 on a half day), no live-only feeds in backtests, decide at the close of T and fill at the open of T+1. | Look-ahead fixes in TradingAgents; FINSABER; "Profit Mirage" and Look-Ahead-Bench (5.1, 7.1) |
| P4 | **Independent first, aggregate statistically, debate only on disagreement.** | "Debate or Vote" (NeurIPS 2025); Lorenz 2011; Good Judgment aggregation (3.2) |
| P5 | **Information diversity, not persona diversity.** Analysts get different data slices. Optional model diversity. No famous-investor personas. | Hidden-profile research (Stasser and Titus); persona prompting shows no accuracy gain (3.2) |
| P6 | **Authentic dissent.** The challenger is the analyst whose own sealed view most opposed the pool. | Nemeth 2001 (3.2) |
| P7 | **Score everything; calibrate; weight by track record only with shrinkage.** Each view is a resolvable forecast with a Brier score. Weights stay equal until a minimum sample, then tilt with shrinkage, a floor and a cap; the equal-weight pool is always logged beside the weighted one. Each analyst's probabilities pass through a calibration map fitted only on its resolved forecasts. | Superforecasting and Good Judgment track records; Bridgewater believability; ai-hedge-fund v2's clipped scorecard weights; LLM overconfidence findings (4, 6.1, 6.2). The evidence that track-record weighting improves *returns* is mixed. ContestTrade's Sharpe collapse from 3.12 to 0.07 removed its research tools as well as its contests, so it is **not** evidence for track-record weighting alone (removing one contest gave 1.78 or 2.01, in a single six-month run). FinPanel found that track records changed nothing: a null result (2.2, 6.2, 9.3 item 8). Hence equal weights by default and shrinkage. |
| P8 | **Separate idea generation from sizing; keep the LLM a bounded overlay.** Sizing is deterministic (fractional Kelly, caps, volatility targeting). By default the LLM pool may move the quant core's probability by at most 0.05, and the bound shrinks when resolved forecasts show the LLM pool ranking worse than the quant. | Best-ideas and sizing studies; full Kelly is ruinous with estimated edges (6.3); Track-or-Follow: text added no rank information and bounding kept it harmless (6.6) |
| P9 | **An invariant risk core.** Drawdown de-risk with re-risk hysteresis, a drawdown freeze that only a human clears, a daily-loss halt, liquidity caps, a restricted list, a hard position cap and a kill switch. Nothing an LLM says can override it. | Multi-manager pod practice; SEC Rule 15c3-5 and Knight Capital; drawdown literature (3.1, 6.4) |
| P10 | **Honest evaluation.** Next-open fills, costs and impact, baselines (benchmark, equal weight, quant-only), signal IC with power analysis, composition-matched random books, cost breakeven, factor attribution, deflated Sharpe with a trial ledger, a buffered pre/post model-cutoff split, a re-identification gate for blind mode, breadth through a screen, and a forward quant-only shadow book. | FINSABER; Bailey and López de Prado; StockBench and DeepFund live results; the six-line report standard (7.1 to 7.4, 9.3) |

## 3. Package layout and module contracts

```
aifirm/
  config.py            FirmConfig (TOML), validation, fingerprint, MODEL_TRAINING_CUTOFFS
  schemas.py           LLM output schemas and internal records (views, aggregates, targets, forecasts, news events)
  universes/           bundled universe lists (sp100.txt) and ticker-file parsing
  data/                cache, http, prices, fundamentals (SEC), macro (FRED), news (feed, cutoff, source tiers), factsheet
  llm/client.py        LLM interface, ClaudeLLM, OfflineLLM, usage and budget, disk cache, CallRecord
  agents/
    roles.py           AnalystRole specs (task, checklist, data slice)
    analysts.py        prompts, data slices, blind mode, run_analyst
    grounding.py       numeric grounding checker, fact-ID resolution
    news_reader.py     quarantined news reader: headlines -> typed, tiered NewsEvents
    tier_gate.py       caps bullish views that rest only on unverified news
    committee.py       run_committee: views -> gate and calibration -> pool or overlay -> challenge -> memo
    risk_review.py     LLM portfolio risk review (can only scale down)
    reflection.py      process lessons from resolved forecasts
    reid.py            re-identification probe of the blind payload
    offline.py         make_offline_llm(): deterministic responders for every role
  quant/
    signals.py         cross-sectional factor model -> the "quant" analyst view
    screen.py          broad-universe screen and the deep-dive set
  portfolio/
    aggregate.py       pool_probabilities, aggregate_views
    overlay.py         bounded overlay on the quant core, adaptive bound
    construct.py       construct_portfolio
    risk_engine.py     RiskState, apply_risk_limits, orders_from_target
  execution/broker.py  PaperBroker (costs, dividends), AlpacaBroker (paper endpoint unless changed in code)
  journal/
    ledger.py          append-only hash-chained JSONL ledger
    forecasts.py       ForecastStore (SQLite), resolution, scorecard, believability_weights
    calibration.py     per-agent Platt calibration, point-in-time rank IC and paired IC
    lessons.py         LessonStore (point-in-time retrieval)
  pipeline.py          gather_data and decide: one decision cycle, shared by live runs and backtests
  books.py             paper-book state files and settlement (shared by the live and shadow books)
  shadow.py            quant-only shadow book and compare_books
  backtest/
    metrics.py         Sharpe, Sortino, drawdown, PSR, DSR, MinTRL, Brier, Newey-West, OLS with HAC errors
    engine.py          walk-forward backtest, baselines, cutoff split, cost replay
    ic.py              forward excess returns, per-date rank IC, paired IC tests, power analysis
    random_books.py    composition-matched random books (the null)
    evaluation.py      cost sensitivity, breakeven search, the Evaluation record
    factors.py         ETF factor attribution
  report.py            markdown decision and backtest reports, six_lines
  cli.py               aifirm factsheet | run | compare | screen | backtest | score | reid-probe | verify-ledger | unfreeze
```

### 3.1 Data layer and shared contracts

**Fact sheets** (`aifirm.data.factsheet`).

* `Fact(id, label, value, unit, as_of, source)`. `FactSheet` holds `ticker`, `as_of` (the
  decision session), `name`, `sector`, `facts`, `events` (dated SEC filing events, text),
  `headlines` (raw, live only), `data_gaps` (told to the agents), and three news fields:
  `news_quarantined`, `news_events` (news fact ID to `NewsEvent`) and
  `quarantined_headlines` (raw text kept for audit, never rendered).
* Methods: `get(id)`, `ids()`, `render(sections)` with sections from
  `("price", "fund", "macro", "events", "news")`, `attach_news_events(events, gaps,
  cite_filings=True)`, `news_tier(id)`, `has_news_content()`.
* Fact IDs: `TICKER.key` for ticker facts (`NVDA.ret_3m`, `NVDA.pe_ttm`, `NVDA.vol_60d`,
  `NVDA.mom_12_1`, `NVDA.roe`, `NVDA.earnings_yield`, `NVDA.fcf_yield`,
  `NVDA.operating_income_margin`, `NVDA.adv_20d`, `NVDA.price`), `MKT.<benchmark>_*` and
  `MACRO.<FRED id>` for shared facts. `PRICE_KEYS` lists the price-derived keys.
* **Event facts** exist only after `attach_news_events`: `TICKER.news_N` is a
  reader-extracted news event and `TICKER.filing_N` a dated filing event made citable.
  Their value is text and their unit is `"event"`. A news fact's label carries its tier,
  type, direction and materiality and no numbers; its source is the code-assigned source
  name and tier, never the raw publisher string. Once a sheet is quarantined, `render()`
  never prints headline text, even if `headlines` is filled again.
* `build_factsheet(ticker, as_of, prices, fundamentals, macro, filings, headlines, sector,
  benchmark)` re-slices prices at `as_of` itself.

**Prices** (`aifirm.data.prices`). `PriceData` has `close`, `adj_close`, `volume`, `open`,
`raw_close` and `splits` frames (dates by tickers).

* `close` is Yahoo's close, back-adjusted for splits up to the download date. It is one
  consistent basis, so it is used for fills, share accounting, dollar volume and marking.
* `raw_close` is the **as-traded** close, rebuilt from `close` and the split ratios. It is
  used for every price *level* shown to agents (the price fact, market cap, multiples) and
  for the minimum-price rule, so a level matches an as-filed share count.
* `adj_close` (dividend- and split-adjusted) is used only for returns.
* `splits` holds split ratios by ex-date. `price_level(t)` returns the as-traded series;
  `split_ratio(t, after, through)` puts an older share count on the as-of share basis.
* `as_of(date)` keeps rows and split events on or before the date. `load_prices` drops
  every row after `last_completed_session()` (a weekday whose close is final after 16:30
  New York time) and never caches a partial bar; a download in which a ticker came back
  empty is cached for 30 minutes only.

**Fundamentals and filings** (`aifirm.data.fundamentals`).

* `load_fundamentals(ticker, as_of, cik=None)` reads SEC XBRL "companyfacts" as filed.
  A filing counts as known at the as-of close only if its EDGAR acceptance time is before
  that session's New York close (`decision_cutoff`: 16:00, 13:00 on a half day); with no
  acceptance time, a filing dated D is known
  from the next day. `Fundamentals.gaps` lists what is missing. `foreign_filer` marks
  20-F/40-F and IFRS filers, for which no market-cap-based multiples are computed.
* `recent_filings(ticker, as_of, lookback_days, cik=None)` lists 8-K items, 10-Q/10-K
  and Form 4 counts that were public by the as-of close. It raises when the history
  cannot be loaded, so the caller records a data gap.
* The SEC filer is resolved per date (`TICKER_CIK_HISTORY` lists known predecessors). A
  filer can be pinned with `universe.ciks` in the config, `set_cik_overrides()` or the
  `AIFIRM_CIK_OVERRIDES` environment variable.

**Macro** (`aifirm.data.macro`). `macro_snapshot(as_of)` reads FRED's CSV. Daily market
series get a one-day availability lag. Monthly series appear only after a conservative lag
(CPI 50 days, unemployment 46 days after the observation date) and after any release date
listed in `DELAYED_RELEASES` (for example the 2025 shutdown). The value shown is the latest
revision, not the first print.

**News** (`aifirm.data.news`). `Headline(title, publisher, published, summary, link)`.
`recent_headlines(ticker, as_of, max_items=12, max_age_days=14)` reads Yahoo's feed (live
only; it cannot be replayed).
`decision_close_utc(as_of, close_time=None)` is the decision instant: the session's own
New York close, `aifirm.data.calendar.session_close_time(as_of)` (13:00 on the scheduled
half days, else 16:00), unless `close_time` ("HH:MM") is given; `recent_headlines` and
`headlines_as_of` take the same `close_time`. `headlines_as_of(headlines, as_of)` keeps
only headlines with a parseable publish time at or before that instant, newest first. `source_tier(publisher, link,
issuer_domains)` assigns the tier (section 3.14).

**LLM interface** (`aifirm.llm.client`). `LLM.structured(role=, model=, system=, user=,
schema=, effort=, max_tokens=) -> (obj, CallRecord)`. It raises `LLMCallFailed` and
`BudgetExceeded`; callers turn a failure into an abstention, a data gap or a held book,
never into a default position. `OfflineLLM(responders)` dispatches on the longest matching
role prefix. Role strings: `analyst.<name>`, `challenge`, `revision.<name>`, `memo`,
`risk_review`, `lesson`, `news_reader`, `reid_probe`. Identical calls are cached on disk
(`$AIFIRM_CACHE_DIR`, else `$XDG_CACHE_HOME/aifirm`, else `~/.cache/aifirm`).

**Schemas** (`aifirm.schemas`): `AnalystView`, `Scenario`, `EvidenceItem`, `Challenge`,
`Revision`, `ICMemo`, `KillCriterion`, `ChecklistItem`, `RiskReview`, `RiskConcern`,
`Lesson`, `ViewRecord`, `AggregatedView`, `TargetPosition`, `TargetPortfolio`, `Order`,
`Fill`, `ForecastRecord`, `NEWS_EVENT_TYPES`, `ReaderEvent`, `NewsReading`, `NewsEvent`,
`ReidGuess`, `clamp`, `sanitize_view`.

**Config** (`aifirm.config`). `FirmConfig` and its sections (`universe` with `screen`,
`models`, `analysts`, `process`, `portfolio`, `risk`, `costs`, `execution`, `calibration`,
`evaluation`). Unknown keys are errors, numbers are range-checked, and cross-checks run
(for example `portfolio.max_position <= risk.max_position_hard`, and overlay mode needs
the quant analyst). A relative `runs_dir` or screen file is relative to the config file
(`runs_path()`, `screen_file_path()`). `fingerprint()` hashes the config without
`runs_dir` and `evaluation`, so evaluation settings do not count as a new trial, with one
exception: a non-default `evaluation.cutoff_buffer_days` is hashed (it decides which LLM
forecasts count as clean track record, so it can change a decision). With the screen on,
the screen's sources (`preset`, `file`, inline `tickers`) are replaced in the hash by the
resolved universe (`investable_tickers()`). `with_frozen_universe()` returns a copy whose
screen sources are that resolved list (same fingerprint and `universe_digest()`);
`run_backtest` uses it so a long run reads the universe file once.
`MODEL_TRAINING_CUTOFFS` holds the vendor training cutoff per model; `model_cutoff(model)`
reads it. Python 3.11 or later is needed (the loader uses `tomllib`).

### 3.2 `agents/roles.py`

```python
@dataclass(frozen=True)
class AnalystRole:
    name: str                    # "fundamental" | "technical" | "macro" | "events"
    title: str
    mandate: str                 # what this analyst is responsible for judging
    checklist: tuple[str, ...]   # concrete things to evaluate (task-specific, not persona)
    sections: tuple[str, ...]    # FactSheet.render sections this analyst sees (its data slice)
    extra_fact_keys: tuple[str, ...] = ()  # ticker facts from other sections it may also see
ROLES: dict[str, AnalystRole]
```

| Role | Sections | Extra facts |
|---|---|---|
| fundamental | `fund` | `price`, `ret_12m` |
| technical | `price` | none |
| macro | `macro` | `beta_1y`, `vol_60d`, `ret_3m`, `rel_ret_3m` |
| events | `events`, `news` (filing facts, news-event facts) | `ret_1m`, `price`, `vol_20d` |

Every slice's header shows the sector label. News-event and filing facts reach only the
events slice.

### 3.3 `agents/analysts.py`

```python
def analyst_system_prompt(role, horizon_days, benchmark) -> str     # stable text: prompt-cacheable
def analyst_user_prompt(role, fs, horizon_days, benchmark, lessons, blind) -> str
def run_analyst(llm, role_name, fs, cfg, model, sample=0, lessons=None) -> ViewRecord
def role_slice(role, fs) -> FactSheet;  shown_sheet(role, fs, blind) -> FactSheet
def blind_factsheet(fs, alias="STOCK") -> FactSheet
```

* Prompts use neutral framing, give calibration guidance (honest one-month probabilities
  usually sit between 0.35 and 0.65), require fact-ID citations, forbid numbers from
  memory, allow abstention, and say that tier-C news is an unconfirmed claim that must not
  raise a forecast. They do not spell out what the tier gate accepts (a prompt that teaches
  the model how to pass the gate invites it to cite an unrelated fact beside a rumor).
* `run_analyst` turns `LLMCallFailed` or `BudgetExceeded` into an abstaining view with
  `grounding_issues=["llm failure: ..."]`. It applies `sanitize_view`, resolves citations,
  runs the grounding check and sets `grounding_ok`. `ViewRecord.model` is the model that
  actually served the call (a server-side fallback may answer instead of the requested
  model), so the forecast is scored and cutoff-checked against the right model.
* **Blind mode** (`process.blind_mode`, default off) replaces the ticker with `STOCK` (also
  inside fact IDs), removes the company name, drops USD-scale facts and the share count,
  makes dates relative to day T, withholds headlines and news-event facts, and scrubs the
  ticker, name, SEC filer names, CIKs and URLs from every piece of free text, including
  the sector label, data gaps and filing events. The sector label itself is kept, and on
  real data it can be very identifying (section 3.15).

### 3.4 `agents/grounding.py`

```python
def check_view(view, fs, rel_tol=0.03, prompt_numbers=()) -> GroundingReport
def resolve_fact_id(fid, known, ticker="") -> str
```

* `fs` is the slice the analyst was shown, so a cited ID it never saw is unknown.
* Matching is **unit-aware**: a percentage must match a percent fact, a dollar amount a
  dollar fact.
* Matching is **scoped by precision**. A number is checked leniently (rounding, 3%
  relative, 0.05 absolute) against the facts the view cites. Against the rest of the slice
  it must be written with informative precision (at least three significant digits, or
  one decimal and two significant digits) and equal the fact at that precision. So an
  invented round number ("7%") cannot pass by matching an uncited fact.
* Numbers that appear in sheet text the analyst was shown (labels, filing events) count as
  supported. The view's own forecast numbers, the base rate (0.5 or 50%), years and counts
  of 12 or less before a time unit ("3 months") are ignored.
* **Citation rule.** A non-abstaining view that states numbers but cites no fact, or cites
  nothing although its slice holds only numeric facts, is flagged.
* `resolve_fact_id` accepts a bare key (`ret_3m`) or a pattern (`MKT.*_vol_20d`) that names
  exactly one fact on the sheet.
* `ok` is false if any cited ID is unknown, more than two numbers are unsupported, or the
  citation rule fails. A flagged view is kept but its pooling weight is halved.

### 3.5 `agents/committee.py`

```python
@dataclass
class CommitteeResult:
    ticker: str
    views: list[ViewRecord]            # RAW stated views (what the forecast store records)
    aggregate: AggregatedView          # the final probability the firm sizes from
    challenge: Challenge | None
    challenger: str | None
    memo: ICMemo | None
    edge_multiplier: float
    hold_current: bool = False         # a held name whose required memo failed
    llm_aggregate: AggregatedView | None = None   # pool of the LLM analysts alone
    pool_views: list[ViewRecord]       # tier-gated, calibrated copies used for pooling
    tier_gate_notes: list[str]
def run_committee(llm, fs, cfg, *, quant_view, believability, lessons, held, log=..., calibrators=None,
                  max_tilt=None) -> CommitteeResult
```

1. **Sealed views.** Each enabled LLM analyst writes `samples_per_analyst` views
   (models rotate across samples; calls run on a thread pool of `max_parallel_calls`).
   The quant view is appended if given.
2. **Weights.** Each view's weight is its agent's believability divided by that agent's
   number of samples, halved if the view is ungrounded. Abstentions are not pooled.
3. **Pooling inputs** (`pool_inputs`). Each LLM analyst's view goes through the tier gate
   (if `process.tier_gate_enabled`) and then its agent's calibrator. These are copies:
   `views` keep the raw stated probabilities. The quant is neither gated nor calibrated.
   A `pool_inputs` ledger event records raw and pooled probabilities per view.
4. **Final aggregate.**
   * `process.combination = "overlay"` (default): the LLM views are pooled
     (`aggregate_views`) and the result tilts the quant core within the bound
     (section 3.8). The expected excess return used for sizing is derived from the final
     probability on the quant's volatility scale, so the LLMs' own excess-return numbers
     never reach sizing. `dispersion` is the LLM pool's.
   * `"pool"`: every member, quant included, is one member of the trimmed log-odds pool.
5. **Challenge round.** If `challenge_enabled`, the dispersion of *all* members (quant
   included) exceeds `challenge_dispersion_threshold`, and at least two views are active,
   the challenger is the LLM analyst whose probability (as pooled, after the gate and
   calibration) is farthest from the members' pool on the opposite side of 0.5 (if none,
   the farthest overall). It writes a `Challenge` (role `challenge`) from the full fact
   sheet and an anonymised summary of the pool. Each other LLM analyst writes a private
   `Revision` (role `revision.<name>`) from its own slice, its own sealed view and the
   challenge. A claimed change that cites no
   fact from its slice or the challenge is rejected. Accepted revisions are gated,
   calibrated and re-pooled; `flips` counts analysts who crossed 0.5.
6. **Memo.** With `memo_enabled`, the chair (role `memo`, `committee_model`,
   `committee_effort`) must review a name when it is held, when the final probability
   clears `portfolio.min_prob_to_hold` (or its short mirror), or when
   `|p − 0.5| ≥ memo_min_edge`. Views are shown anonymised (letters, in a ticker-seeded
   order, role words removed); in overlay mode the memo also shows the quant core, the
   LLM pool and the tilt. `edge_multiplier` is the memo's conviction multiplier clamped to
   [0, 1]. It is 0 for `review`, for `avoid`/`sell` in a long-only book, and for `hold` on
   a name the book does not own. A name that needed no memo and is not held gets 0 (an
   unreviewed edge never opens a position). With memos disabled the multiplier is 1.0.
7. **Fail closed.** A failed analyst call is an abstention, a failed challenge keeps the
   sealed pool, a failed revision keeps the sealed view. A failed memo on an unheld name
   gives edge 0. A failed memo on a held name sets `hold_current=True` (edge 1.0); the
   pipeline then keeps that name at its current weight
   (`construct_portfolio(keep_current=...)`) and warns.
8. Events `view`, `pool_inputs`, `aggregate` (sealed and revised), `challenge`, `revision`
   and `memo` go to `log(event, payload)`; the pipeline replays them in a deterministic
   order into the ledger.

### 3.6 `agents/risk_review.py`, `agents/reflection.py`, `agents/offline.py`

```python
def review_portfolio(llm, target, factsheets, cfg) -> RiskReview | None
def apply_risk_review(target, review, current_weights=None, benchmark=None) -> TargetPortfolio
def write_lesson(llm, forecast, view_summary, cfg) -> Lesson | None
def make_offline_llm() -> OfflineLLM
```

* The risk reviewer's multipliers are clamped to [0, 1]; several concerns on one name use
  the smallest; a portfolio-level concern multiplies every name. A multiplier `m` scales
  the leg's **plan**, not the partial step (`reviewed_weight`): `ideal_weight` becomes
  `m × ideal` and the leg is held to `m × max(|ideal|, |step|)`. On a leg at or above its
  plan (a reduction, an exit) that is `m` times the step; on a leg still building toward
  its plan it caps the leg at `m` times the plan. Multiplying the step every cycle
  compounded with partial rebalancing: a standing x0.5 at `rebalance_fraction` 0.5 settled
  at a third of the plan instead of a half. Holds and hand-built legs are multiplied
  directly. The ledger note shows each leg's weight before and after. **A halt means no trades
  this cycle**: given `current_weights`, the result is the current book. A halt without
  `current_weights` raises (any guess would be a trade).
* Lessons are written for the most informative newly resolved forecasts of LLM analysts
  and the chair (at most 8 per cycle), from the view as recorded in the ledger.
* The offline responders apply simple, transparent rules to the rendered fact lines and
  cite real fact IDs. The `news_reader` responder classifies by keywords and never copies
  headline text. The `reid_probe` responder guesses uniformly at random, seeded by the
  prompt, so its hit rates sit at chance. Offline output is not AI.

### 3.7 `quant/signals.py` and `quant/screen.py`

```python
def factor_scores(factsheets) -> dict[str, dict[str, float]]
def quant_views(factsheets, horizon_days) -> dict[str, ViewRecord]   # agent="quant", model="deterministic"
def horizon_sigma_pct(fs, horizon_days) -> float                     # vol_60d (else 30%) x sqrt(h/252), in %
def screen_universe(factsheets, quant, prices, as_of, *, top_n, min_dollar_volume, min_price, held=()) -> ScreenResult
```

**Factor model.** Factors are cross-sectional z-scores (population std-dev, winsorised at
±3): momentum (`mom_12_1`), low volatility (−`vol_60d`), quality (`roe`,
`operating_income_margin`) and value (`earnings_yield`, `fcf_yield`). A two-input factor
averages its inputs' z-scores and is z-scored again. The composite is the mean of the
factor z-scores a name has; missing data are left out. The probability mapping is
deliberately modest: `p = 0.5 + 0.06·tanh(composite)`, so `p` stays in [0.44, 0.56].
Expected excess return is `σ_h·Φ⁻¹(p)`. A name with no scorable factor abstains; with
fewer than three names the quant abstains on every name.

**Screen** (`universe.screen`, off by default). The universe is `universe.tickers` plus a
bundled preset (`sp100`, 100 current S&P 100 members), a ticker file and inline tickers.
At each decision:

1. Fact sheets are built for every name from prices, SEC data and macro data, without any
   LLM call. (In a live run `gather_data` also fetches headlines for every name; only the
   deep-dive names' headlines are read, the rest are withheld unread.)
2. The quant scores the whole cross-section.
3. A name is **eligible** if the quant has a view, its last close on or before the decision
   is at least `risk.min_price` and at most 5 calendar days old, and its 20-day average
   dollar volume (with at least 5 valid days) is at least `min_dollar_volume` (default
   10 million USD). Prices are cut at the decision close inside the screen. The minimum
   price is tested on the **as-traded** close (`PriceData.price_level`), like the risk
   engine's price fact: Yahoo's `close` is back-adjusted for every split up to the
   download date, so it would screen out a name on a split that had not happened yet.
   Dollar volume uses `close × volume`, which splits do not change.
4. Eligible names are ranked by quant probability (ties by ticker).
5. The **deep-dive set** is the top `deep_dive_top_n` eligible names (default 12) whose
   probability is above 0.5, then every held name, whatever its rank. Held names are
   always included (not configurable): leaving one out would force a sale. Keep
   `deep_dive_top_n` at 5 or more: a date's paired IC needs 5 common names, so a narrower
   band can never measure the adaptive overlay bound or the shadow-book ranking test (the
   decision warns, and both say how many dates were skipped for too few names).

Only deep-dive names get LLM calls, news reading and new positions. The quant still writes
a forecast for every screened name, which gives the IC statistics their breadth. The full
ranking goes to the ledger as a `screen` event. The universe *list* is not point-in-time:
a list of current members is survivorship-biased, and backtest reports say so.

### 3.8 `portfolio/aggregate.py` and `portfolio/overlay.py`

```python
def pool_probabilities(probs, weights=None, trim=0.2) -> float
def aggregate_views(ticker, views, trim=0.2, use_revised=False) -> AggregatedView
def combine(p_quant, p_llm, max_tilt) -> tuple[float, float]           # (p_final, tilt)
def overlay_aggregate(ticker, quant, llm_views, *, trim, max_tilt, sigma_h_pct, use_revised=False)
def adaptive_bound(records, as_of, cfg, *, include_pre_cutoff=False, min_names=5) -> OverlayBound
```

**Pooling.** A weighted, trimmed mean in log-odds space (log-odds = `ln(p/(1−p))`).
Probabilities are clipped to [0.01, 0.99]; with at least four members, `trim` of the total
weight is removed from each tail, fractionally at the boundary. Expected excess return is
the weighted median of the members'. Dispersion is the weighted standard deviation of the
member probabilities. `prob_equal_weight` is the same pool with unit weights. With no
active member the result is 0.5 and an excess of 0.

**Bounded overlay** (RESEARCH 6.6):

```
p_final = clip(p_quant + clip(p_llm − 0.5, −max_tilt, +max_tilt), 0, 1)
expected excess used for sizing = σ_h · Φ⁻¹(p_final)
```

* `p_quant` is the quant's probability; when the quant abstains the core is 0.5.
* `p_llm` is the pool of the LLM analysts' tier-gated, calibrated probabilities. When every
  LLM analyst abstains, `p_final = p_quant` exactly.
* The tilt is the LLM pool's **own edge** (`p_llm − 0.5`), a residual added to the core, not
  its distance from the quant. An LLM pool at 0.5 is "no information" and leaves the core
  exactly where it is: a view the tier gate capped at 0.5, an analyst whose calibration map
  shrank it to 0.5 for lack of skill, an analyst stating no edge. (The earlier rule,
  `clip(p_llm − p_quant)`, read such a pool as a vote that the stock would lag and cut every
  quant pick toward 0.5, below `min_prob_to_hold`, so the gate and calibration sold what
  the quant-only shadow held.) A bullish pool can only raise the core, a bearish one only
  lower it.
* `σ_h` is `horizon_sigma_pct` (the quant's own volatility scale).
* With the quant in [0.44, 0.56] and the default bound 0.05, the final probability stays in
  [0.39, 0.61].
* Everything after the overlay can only shrink: the memo, the dispersion shrink, the risk
  review and the risk engine. With the screen, the LLM cannot promote a name from outside
  the deep-dive band.

**Adaptive bound.** The rank IC (information coefficient) of a forecaster on one date is
the Spearman correlation, across names, between its probabilities and the realised excess
returns. At each decision, `paired_rank_ic` compares the per-date rank IC of the
`llm_pool` forecasts with the quant's, on the names both forecast that date (at least 5
common names), using only forecasts resolved on or before the decision date. Newey-West
lags are `ceil(horizon / rebalance_every_days) − 1`. With fewer than `overlay_min_dates`
(12) such dates, or no variation, the configured bound applies. Otherwise, if the
t-statistic of `IC_llm − IC_quant` is below −1 the bound is halved, and below −2 it is 0
(the overlay is switched off). It never rises above `overlay_max_tilt`. The evidence is
logged as an `overlay_bound` event and a warning is printed when the bound is reduced.

### 3.9 `portfolio/construct.py`

```python
def construct_portfolio(aggs, edge_multipliers, factsheets, current_weights, cfg, as_of, sectors,
                        returns=None, keep_current=()) -> TargetPortfolio
```

Per ticker (`h = horizon_days`):

```
sigma_h = sigma_ann · sqrt(h/252)       sigma_ann: vol_60d floored at 5%; without it max(30%, vol_20d, realised vol)
mu_prob = sigma_h · Φ⁻¹(p)              p clipped to [0.01, 0.99]
mu_exp  = expected_excess_pct / 100
mu0     = 0.5 · (mu_prob + mu_exp)
m       = clamp(edge_multiplier, 0, 1)
d       = max(0, 1 − dispersion_shrink · dispersion)
long if p ≥ min_prob_to_hold and mu0·m·d > 0
w       = sign(kelly) · min(max_position, |kelly|) · m · d,   kelly = kelly_fraction · mu0 / sigma_h²
```

* The chair multiplier and the disagreement factor apply **after** the per-name cap, so a
  multiplier of 0.5 always halves the position.
* Then: sector caps, sleeve volatility targeting (sample covariance of the last 126 days
  when returns are given, else a constant correlation of 0.3), the core-satellite split
  (`core_weight` goes to the benchmark), and the gross and cash-floor limits.
* **Two caps.** Sizing caps each name at `portfolio.max_position` (8%). The hard limits the
  final book must satisfy use `risk.max_position_hard` (10%). A holding that drifted
  between the two is not force-traded.
* **Trade path.** A no-trade band (`no_trade_band`, 0.5% of NAV); **partial rebalancing**:
  each trade moves `rebalance_fraction` (0.5) of the way to the target, unless the gap left
  would be smaller than the band, or than `MIN_RESIDUAL_GAP` (0.25% of NAV) when the band
  is smaller (with a band of 0 an exit would otherwise halve for ever), in which case the
  trade completes; and a turnover cap. Each position records `ideal_weight`, the target
  before this trade path, which the risk engine's de-risk caps at one half.
  Turnover is **one-way**: `0.5 · Σ|w_target − w_current|` over every leg, core included.
* **Forced exits.** A held name absent from `aggs` (no longer covered) is sold in full: it
  skips the band, partial rebalancing and the turnover cap, and its turnover is spent first.
  The remainder of an ordinary exit (ideal weight exactly 0) that has drifted under the
  band is completed the same way (`exit_completed`); the band would otherwise keep it for
  ever.
* **No information, no trade.** A held name with `n_views == 0` keeps its current weight. So
  does a name in `keep_current` (a held name whose memo failed). Only a hard limit can cut
  them.
* Cost controls yield to risk limits: if the band or turnover cap would leave a hard limit
  breached, the book moves toward the ideal target just far enough to comply, with a note.
* **Exposure.** Every target carries an `exposure:` note: names held, active gross, core,
  cash, how many names cleared `min_prob_to_hold`, how many sit at the cap and which
  book-level limits bound. Because quarter Kelly on any real edge is far above the cap,
  the active gross is roughly (names that clear the threshold) × `max_position` until the
  volatility target or sector caps bind. On a narrow universe the book is often mostly
  cash; a broad screened universe supplies enough qualifying names.

### 3.10 `portfolio/risk_engine.py` (the invariant core)

```python
@dataclass
class RiskState:
    peak_nav: float
    frozen: bool = False
    last_nav: float | None = None
    last_as_of: str | None = None
    frozen_at_nav: float | None = None     # marks a freeze, so a human reset can be recognised
    trough_nav: float | None = None        # lowest NAV since the last peak
    derisked: bool = False                 # the de-risk episode is in force
    derisk_caps: dict[str, float] = {}     # per leg: the |weight| the de-risk last set in this episode
def update_risk_state(state, nav) -> RiskState
def apply_risk_limits(target, state, nav, prev_nav, current_weights, cfg, factsheets, runs_dir, *,
                      worst_day_return=None) -> tuple[TargetPortfolio, list[str]]
def orders_from_target(target, positions, prices, nav, adv, cfg, benchmark) -> tuple[list[Order], list[str]]
```

Rules, in order:

1. **Kill switch.** If `<runs_dir>/<kill_switch_file>` exists, the target is the current
   book: no orders.
2. Non-finite weights and shorts in a long-only book go to 0. Weights above
   `risk.max_position_hard` are clipped.
3. A human reset after a freeze (`frozen` cleared, `frozen_at_nav` still set) restarts the
   high-water mark at the current NAV.
4. **Freeze.** A drawdown from peak of `drawdown_freeze` (20%) or more sets `frozen`; the
   active sleeve goes to 0 (the core is kept) until a human resets it.
5. **De-risk with hysteresis.** A drawdown of `drawdown_derisk` (10%) or more halves the
   active weights and sets `derisked`. The book stays halved until
   `nav ≥ trough + 0.5 · (peak − trough)` or a new peak (RESEARCH 6.4: re-risk after
   recovering half the drawdown). "Halved" is relative to the un-derisked plan and does
   not compound: a leg built by construction carries `TargetPosition.ideal_weight` (its
   target before the band, partial rebalancing and turnover cap) and is capped at
   `0.5 × |ideal_weight|`; halving the incoming target instead compounded with partial
   rebalancing (a step from a book already halved last cycle) and sold every cycle toward
   a third of the ideal. The plan includes the risk reviewer's cut (it scales
   `ideal_weight` by its multiplier), so a reviewed leg is capped at half of the reviewed
   plan. A deliberate hold (`pipeline:hold` on an LLM outage, `unreviewed_hold`,
   `no_view_hold`, `risk_review_halt`) is halved on the cycle the de-risk triggers. The
   weight the de-risk sets for each leg is recorded in `derisk_caps`, and the same level as
   a share count in `derisk_shares`; on every later cycle of the episode a hold keeps its
   current weight (an LLM outage never forces a sale, and a hold is never sold below the
   recorded shares) and is cut back to the recorded shares only when it holds more shares
   than that by at least the no-trade band, i.e. the earlier sale did not execute (an order
   blocked for a missing price or ADV, or rejected). Judging in shares means a hold whose
   price rose after the sale (same shares, higher weight) is kept; without a price the hold
   is kept. The record is cleared on re-risk, a freeze or a human reset. A
   hand-built target without `ideal_weight` is halved. The ledger message names the legs
   actually cut ("nothing cut" when every leg is already within its de-risk level).
6. Restricted tickers and prices below `min_price` go to 0. A name without a verifiable
   price may not grow.
7. **Daily-loss halt.** If any one-day NAV loss since the previous cycle
   (`worst_day_return`, or `nav/prev_nav`) is `daily_loss_halt` (5%) or more, no weight may
   increase this cycle.
8. Backstop: gross and cash floor.

Apart from the kill switch, every rule only moves a weight toward zero. The pipeline feeds
every daily NAV since the previous decision to `update_risk_state` (the backtest passes
`nav_marks`; the live book rebuilds the path from held shares).

`orders_from_target`: quantity = (target weight × NAV − current value) / price; trades
smaller than `no_trade_band × NAV` are skipped, except a full exit; each order is capped at
`max_adv_participation` (2%) of 20-day dollar volume; a missing price or ADV blocks the
order; sells come before buys.

### 3.11 `execution/broker.py`

```python
class PaperBroker:   # __init__(cash, cfg: CostConfig, state_path=None); save()/load()
    def submit(self, orders, fill_prices, date, adv=None, daily_vol=None) -> list[Fill]
    def credit_dividends(self, per_share, date) -> float
    @classmethod
    def from_snapshot(cls, cash, positions, cfg, state_path=None, *, marks=None, dividends_through=None, ...)
class AlpacaBroker:  # ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY; refuses non-paper endpoints unless allow_live=True
def execution_cost(notional, price, adv, daily_vol, cfg) -> tuple[float, float]
    # slippage bps = slippage_bps + impact_coefficient · 1e4 · daily_vol · sqrt(|notional| / adv)
```

Paper fills happen at the given prices (the next open); buys pay slippage up, sells
receive it down. Buys are scaled down to available cash. Fractional shares are allowed.
`credit_dividends` credits cash per share held; `dividends_through` records the last
settled session so no ex-date is credited twice. `from_snapshot` copies another book's
holdings (used to start a shadow book). `AlpacaBroker` treats any endpoint other than
Alpaca's paper URL as live and refuses it unless constructed with `allow_live=True`, which
the CLI never passes: live trading needs a code change. With `execution.require_approval`
(the default) it submits only after an explicit approval.

### 3.12 `journal/`

```python
class Ledger:                 # ledger.py
    def append(self, event, payload) -> str       # entry = {seq, ts, event, payload, prev_hash, hash}
    def verify(self, anchor=None) -> tuple[bool, int]
class ForecastStore:          # forecasts.py (SQLite)
    def add(self, records) -> list[ForecastRecord]  # returns the records kept
    def all(self) -> list[ForecastRecord]
    def resolve(self, prices, as_of) -> list[ForecastRecord]
    unresolvable: dict[str, str]                    # id -> why a due record could not resolve
def make_forecast_records(run_id, as_of, views, agg, cfg, trading_calendar) -> list[ForecastRecord]
def scorecard(records, as_of=None, include_pre_cutoff=True) -> dict[str, dict]
def believability_weights(records, as_of, agents, min_n=20, prior_n=30, floor=0.25, cap=2.0,
                          include_pre_cutoff=False, min_dates=20) -> dict[str, float]
def fit_calibrators(records, as_of, agents, *, include_pre_cutoff=False, min_n=30, prior_strength=50.0,
                    a_bounds=(-0.5, 0.5), b_bounds=(0.0, 1.5), models=None, min_dates=20) -> dict[str, Calibrator]
def calibrate_view(v, cal) -> ViewRecord
def resolved_rank_ic(records, as_of, agent="committee", *, include_pre_cutoff=False, min_names=3, nw_lags=0)
def paired_rank_ic(records, as_of, agent_a, agent_b, *, include_pre_cutoff=False, min_names=3, nw_lags=0)
    # -> PairedIC(mean_diff, t_stat, n_dates, mean_a, mean_b, n_thin): n_thin = dates skipped for too few names
class LessonStore:            # lessons.py (JSONL)
    def retrieve(self, agent, as_of, k=5) -> list[str]   # only lessons resolved on or before as_of
```

**Ledger.** `hash = SHA-256(prev_hash + canonical JSON of the entry)`. Editing, deleting,
inserting or reordering a line breaks the chain. Deleting the tail is caught by comparing
with a head hash recorded elsewhere (`ledger_anchors.jsonl`). A damaged last line stops
the run.

**Forecast records.** Per committee name: one record per non-abstaining analyst view
(its **raw, sealed** probability), one `committee` record (the final probability the firm
sized from) and, when an LLM pool exists, one `llm_pool` record (the pooled LLM
probability before the overlay, scored on probability only). With the screen, and when the
quant sits on the committee, the quant also gets a record for every screened name outside
the deep dive. The shadow book adds `shadow` records (section 3.16).
`POOLED_AGENTS = {committee, llm_pool, shadow}` are not analysts and are excluded from
pooled analyst statistics.

**Resolution.** Outcome = 1 if the ticker's adjusted-close total return from the `as_of`
close to the `resolve_on` close (horizon trading days later) strictly beats the
benchmark's. A missing print is bridged by the last close on which both series have a
price, at most 5 calendar days earlier. A record that still cannot resolve stays pending
and `unresolvable` says why.

**One forecast per question.** A question is (decision date, ticker, agent, sample)
(`question_key`). `add` keeps the first forecast of a question, so a forced re-decision
cannot count twice. `pre_cutoff` is set when the model may have seen the period in
training: on or before its cutoff plus `evaluation.cutoff_buffer_days` (31), the same rule
as the backtest's `contamination_end` (deterministic code never; an unknown cutoff always).
A non-default buffer therefore changes which track record counts, and the fingerprint.

**Believability** (P7). The Brier score is the mean squared error of a probability
forecast against the 0/1 outcome; always saying 0.5 scores 0.25.
`skill = (0.25 − brier) / 0.25` and `weight = 1 + n/(n + prior_n) · 4 · skill`, clipped to
[0.25, 2.0], with `prior_n = 30`. The weight is exactly 1.0 until the agent has at least
`min_n` (20) distinct resolved questions **and** `believability_min_dates` (20) distinct
decision dates (one date with 100 names is one market episode, not a track record). Only
records with `resolve_on ≤ as_of` count; pre-cutoff LLM records are excluded unless the run
is offline or quant-only.

**Calibration** (`journal/calibration.py`; config section `calibration`). Per LLM analyst,
a Platt map in log-odds space:

```
logit(p_cal) = a + b · logit(p_raw)
```

* **Point-in-time.** Fitted at each decision only from that agent's records with
  `resolve_on ≤ as_of`, with the same pre-cutoff rule as believability, and only records
  made by the current `models.analyst_models` (a model change starts a new record).
* **One question, one vote.** k samples of one question weigh 1/k each; a question re-run
  by another run counts once.
* **Prior.** The fit is a maximum a posteriori estimate whose prior is worth
  `prior_strength` (50) forecasts at the identity map: pseudo-observations at the agent's
  own stated probabilities, with outcomes equal to those probabilities. With n questions
  the estimate moves about n/(n + 50) of the way from the identity to the unregularised
  fit (37.5% at n = 30).
* **Bounds.** `b ∈ [0, 1.5]` (shrink or mildly sharpen, never invert the order) and
  `a ∈ [−0.5, 0.5]`.
* **Minimum sample.** Below `min_n` (30) questions, or with questions from fewer than
  `calibration.min_dates` (20) distinct decision dates, the map is the identity: one date's
  questions share one market episode, and the intercept would learn that episode's base
  rate of beating the benchmark rather than the agent's bias.
* **No new direction.** `apply()` leaves a raw 0.5 at 0.5 and returns 0.5 when the map
  would cross 0.5; output is clipped to [0.02, 0.98].
* `calibrate_view` calibrates a *copy* for pooling and scales its expected excess return by
  `Φ⁻¹(p_cal)/Φ⁻¹(p_raw)` (clipped to [0, 1.5]). Stored forecasts keep the raw
  probability, so the map is never fitted on its own output. The quant is never
  calibrated. Fitted maps are logged (`calibration` event) and shown in the decision
  report. Reported Brier scores are in-sample.

**Rank IC.** `resolved_rank_ic` gives the mean per-date Spearman correlation between
probabilities and realised excess returns from resolved records, with a t-statistic
(Newey-West when horizons overlap). `paired_rank_ic` compares two agents on the names both
forecast each date; it drives the adaptive overlay bound and the shadow-book verdict.

### 3.13 `pipeline.py`, `backtest/engine.py`, `report.py`, `cli.py`

```python
def gather_data(cfg, as_of, live, prices=None, *, fundamentals_loader=None, macro_loader=None,
                filings_loader=None, headlines_loader=None) -> tuple[PriceData, dict[str, FactSheet]]
def decide(cfg, llm, as_of, prices, factsheets, *, current_weights, nav, prev_nav, risk_state,
           forecast_store, lesson_store, ledger, run_id, positions=None, trading_calendar=None,
           runs_dir=None, nav_marks=None) -> Decision
def run_backtest(cfg, start, end, mode, llm=None, prices=None, fundamentals_loader=None, macro_loader=None,
                 filings_loader=None, *, run_id=None, factor_prices=None, progress=None) -> BacktestResult
```

`progress(line)` (optional) gets one line per decision of the strategy and of the quant-only
baseline and one before the evaluation; `aifirm backtest` prints them on stderr.

**`gather_data`** resolves `as_of` to the last session in the price index (a holiday means
the session before) and loads fundamentals, filings, macro and headlines as of that
close. Headlines load only when `live` and `process.news_enabled`; gather_data never calls
an LLM. A failed loader is a data gap on the sheet; a ticker with no prices gets no sheet;
a missing benchmark raises.

**`decide`** slices prices at `as_of` itself and runs, in order:

1. The quant over the whole universe; the screen (if on) picks the deep-dive set.
2. News quarantine (section 3.14) on copies of the sheets; fact sheets and the screen go to
   the ledger.
3. Forecasts due by `as_of` are resolved; lessons are written; believability, calibration
   maps and (overlay mode) the adaptive bound are computed point-in-time.
4. The committee runs per deep-dive name.
5. **Outage rule.** If half or more of the LLM analyst calls, or of the required memos,
   failed, or the LLM budget is exhausted, the book is held unchanged (still subject to the
   risk engine). A news-reader failure never counts toward this rule.
6. `construct_portfolio` (with `keep_current` for held names whose memo failed), the LLM
   risk review, then the risk engine, fed with the NAV path since the previous decision.
7. Orders sized at the close, to fill at the next open.
8. Forecast records are stored and a `cycle_end` event records usage and warnings.

`Decision` carries the committee results, targets at each stage, orders, forecasts,
usage, warnings, data gaps, `screen`, `overlay_bound`, `calibrators` and `news`, plus
`exposure()` and `final_probabilities()`. Any unexpected error writes a `cycle_error`
entry and stops the cycle with no orders.

**`run_backtest`** replays `decide` day by day. Modes: `quant` (the quant analyst alone, no
LLM), `offline` (offline rules, not AI) and `llm`. It decides at the close of every
`rebalance_every_days`-th trading day and fills at the next open with costs. It marks the
book every day at closes that are not adjusted for dividends and credits each ex-date's
cash dividend, inferred from adjusted versus unadjusted closes
(`pipeline.implied_dividends`). Baselines run through the same engine: benchmark buy-and-hold
(same total-return accounting, no costs), an equal-weight universe (same dates and costs)
and, except in quant mode, the quant-only configuration. Fact sheets are built once per
decision date and shared by the strategy and the quant-only baseline (in a backtest they
depend only on the date, the prices, the loaders and `cfg.universe`; each policy gets a
deep copy). Each run appends to `<runs_dir>/trials.jsonl` (with `universe_digest` and, for
offline and llm runs, the `prompts` digest); the number of distinct (fingerprint, mode,
start, end, prompts) trials feeds the deflated Sharpe. The fingerprint covers the
*resolved* screened universe (`FirmConfig.investable_tickers`), not the path of the
universe file: editing the file is a new trial, and the same tickers reached through
another path are the same trial. The universe is resolved once when the run starts
(`FirmConfig.with_frozen_universe`): the fingerprint, the prices, every decision and the
trial row's `universe_digest` see the same names even if the file is edited during the
run. Returns are split at `contamination_end` (training cutoff plus
`evaluation.cutoff_buffer_days`). After the run, `result.evaluation` is computed from the
recorded decisions with no new decision and no LLM call (section 3.17).
`BacktestResult` also records daily `exposure`, `baseline_nav`, `universe_note`
(survivorship), `overlay_value()` (value added over the quant-only baseline net of LLM
cost) and `exposure_summary()`.

**`report.py`** renders the decision report (summary, screen, committee with overlay
columns, tier-gate notes, calibration maps, believability, views, memos, portfolio
stages, risk review, risk engine, orders, track record, usage, warnings, shadow book) and
the backtest report (six lines, performance against baselines, significance, "Is there
skill, and would we know?", LLM value added, cutoff split, activity and costs,
scorecard, honest interpretation, warnings). `six_lines(result)` returns the six lines.

**`cli.py`** commands: `factsheet`, `run`, `compare`, `screen`, `backtest`, `score`,
`reid-probe`, `verify-ledger`, `unfreeze`. Run `aifirm <command> --help` for flags.

* Only completed sessions may be used (`--as-of`, `backtest --end`, `score --as-of`).
* A book never decides twice on the same session or on an earlier one; `run --force`
  overrides this and is ledgered. Queued orders waiting for their open are never doubled.
* Strategy flags on `run` and `backtest` (`--preset`, `--universe-file`, `--deep-dive`,
  `--min-dollar-volume`, `--combination`, `--max-tilt`, `--no-calibration`,
  `--no-news-reader`, `--no-tier-gate`, `--rebalance-fraction`) change the fingerprint.
  Evaluation flags on `backtest` (`--random-books`, `--seed`, `--sector-matched-books`,
  `--cost-multipliers`, `--breakeven-cap`, `--ic-min-names`, `--no-factor-proxies`) do
  not; `--cutoff-buffer-days` does only when it differs from the default 31 (it also sets
  which LLM forecasts are flagged `pre_cutoff`). Every flag is validated like the config file.
* `runs_dir` is relative to the config file, so a scheduled run from another directory
  finds the same book and kill switch.

### 3.14 Quarantined news reader and tier gate

**Data flow.**

```
recent_headlines (live only, published ≤ the session's New York close: 16:00, 13:00 on a half day)
  └─► read_headlines (deep-dive names only; news reader enabled; an LLM present)
        1. drop late or undated headlines; keep the newest news_max_headlines (20)
        2. tier each headline by code: source_tier(publisher, link, issuer_domains)
        3. withhold headlines that look like instructions to a model; code records them as tier C events
        4. read tier A/B and tier C headlines in separate calls (role news_reader, no tools, numbered data blocks)
        5. validate: drop events for headlines not shown in that call, at most 2 events per headline,
           sanitise summaries, force rumors to tier C, attach publisher, link host, time and SHA-256 by code
  └─► FactSheet.attach_news_events: TICKER.news_N facts (events slice only), TICKER.filing_N facts (tier A),
      raw headlines moved to quarantined_headlines; render() never shows them again
Every other sheet with headlines (outside the deep dive, reader disabled, no LLM): no events + a data-gap note
```

* **Fail closed.** Any reader failure returns no events and a data gap. It is not counted
  as an LLM outage. With no events, news cannot raise exposure.
* **Summaries** are sanitised by code: no URLs, no tokens containing digits, no
  instruction-like clauses, no markup, at most 25 words.
* The reader never sees the publisher or the link; code judges the source.
* The instruction detector is a set of regular expressions. It will miss adaptive attacks;
  the guarantees come from code-set tiers, the separate calls, sanitisation and the gate.
* **Invisible characters are deleted** before detection and before any text goes into a
  reader block: control (Cc), format (Cf: zero-width characters, bidi controls, the soft
  hyphen, the Unicode tag characters used for "ASCII smuggling"), private-use, unassigned
  and the other default-ignorable code points (U+034F, variation selectors, Hangul
  fillers). Deleting rather than spacing them keeps "dis​regard" detectable as
  "disregard". The detector also matches a spaced reading, and an invisible character
  between two letters or digits (used in place of the spaces between words, it would make
  "Ignoreallpreviousinstructions" one word no pattern sees) or any bidi embedding,
  override or isolate control is itself a suspected injection: the headline is withheld
  from the reader and recorded as tier C.
* **Point-in-time close.** The cutoff is the session's real close
  (`aifirm.data.calendar.session_close_time`): 13:00 on the scheduled half days (the day
  after Thanksgiving, Christmas Eve and 3 July from Monday to Thursday), else 16:00. The
  same close bounds SEC filings (`fundamentals.decision_cutoff`).
* Blind mode withholds news-event facts.

**Tier table** (`aifirm.data.news.NEWS_SOURCES`; the tier is set by code, never by a model):

| Tier | Meaning | Sources |
|---|---|---|
| A | primary source | SEC EDGAR (sec.gov); Business Wire, PR Newswire, GlobeNewswire; the issuer's own site, only for domains listed in `universe.issuer_domains` |
| B | established news organisation | Reuters, Bloomberg, Associated Press, The Wall Street Journal, Financial Times, CNBC, Barron's, MarketWatch, The New York Times |
| C | unverified | everything else: unknown publishers, aggregators, blogs, any rumor event, any attribution the link does not confirm, no usable link |

Matching trusts the **link host** over the free-text publisher name:

1. A link on a table entry's domain (or a subdomain) gets that entry's tier, unless the
   publisher names a *different* entry (tier C).
2. A link on a syndication host (`finance.yahoo.com`) takes the tier of the publisher named,
   matched exactly against the table's aliases; an unknown publisher there is tier C.
3. A link on any other host is tier C, whatever the publisher claims ("Reuters" on
   `reuters-markets.biz` is a spoof).
4. No http(s) link: tier C.

The link host is the one a browser would load (`link_host`): a backslash is a path
separator in an http(s) URL, as in the WHATWG URL standard (so
`https://evil.example\@www.reuters.com/x` is on `evil.example`, not Reuters, although
`urllib.parse.urlsplit` alone reads `reuters.com`), userinfo is not the host, a non-ASCII
host is encoded as a browser encodes it (UTS #46 non-transitional, with the `idna` package:
`busineßwire.com` is `xn--businewire-b4a.com`, not Business Wire, and a zero-width joiner
makes a host invalid), and a host that is not a plain DNS name after encoding is tier C.

This simplifies the research's rule (RESEARCH 5.5): there, tier B needs two
upstream-independent outlets, and newswires carry a known hoax risk. Here one established
outlet is tier B and newswires are tier A. There is no corroboration and no tier-C hold.

**Tier gate** (`agents/tier_gate.py`). It applies only to sheets with news content. A view
is *bullish* if its probability is above 0.5 **or its expected excess return is above 0**
(a 0.5 view with a large positive excess feeds the pooled excess and so the size). The
question is what the view would be without the unverified news, and taint is judged on the
**whole view**: an analyst can always put a rumor and an unrelated fact in two separate
points, so a bullish sealed LLM view passes only if

* **no positive or neutral evidence item** cites an inadmissible event: a tier-C news fact,
  a news fact without metadata, an event-like ID that is not on the sheet (in any letter
  case), or a wildcard citation that matches a tier-C news fact. Only a *negative* item
  (a point against outperformance) may cite unverified news without capping the view; and
* at least one **positive** item cites an admissible fact on the sheet: an *informative*
  filing fact `TICKER.filing_N` (an 8-K reporting results, item 2.02, a material agreement,
  1.01, or a completed acquisition or disposition, 2.01, or an activist SC 13D stake), a
  news fact `TICKER.news_N` of tier A or B, or (every role except `events`) a directional
  non-event fact. A routine filing (a 10-Q or 10-K, a proxy, the Form 4 count, an officer
  change, a passive 13G) only says that something was filed, so it never counts; an SEC
  filing never taints either. Context levels are not directional and never count: the
  share price, `vol_20d`, `vol_60d`, `adv_20d`, `shares_out`, `market_cap`, `MACRO.VIXCLS`
  and `MKT.<benchmark>_vol_20d`. The `events` analyst's only non-event facts are the price,
  the 1-month return and the 20-day volatility, which any rumor-based view could cite, so
  its bullish view needs an informative filing or a tier A/B news event.

A **revision** that raises the probability or the excess above the gated sealed values,
and is bullish, passes only if (0) the reviser's own sealed view did not lean on
unverified news (no positive or neutral item of it taints: the reviser has read the rumor,
so a token revision citing another fact from its slice could still rest on it); (1) the
challenge it answered did not lean on unverified news: none of the IDs cited by the challenge's positive or neutral items
(`ViewRecord.challenge_fact_ids`, stored by the committee for every reviser) taints,
whatever the reviser itself cited; and (2) the IDs the reviser cited as new evidence
(`ViewRecord.revision_fact_ids`, stored when the committee accepts the revision) contain
admissible support and no inadmissible event. Otherwise the revised values are capped at
`max(gated sealed p, 0.5)` and `max(gated sealed excess, 0)`. A capped sealed view is pooled
as a copy: probabilities `min(p, 0.5)` and expected excess returns `min(x, 0)`. The gate
never raises a probability, and in the overlay a capped view at 0.5 tilts the quant core
by 0. The prompts tell the analysts that tier-C news is an unconfirmed claim but do not
spell out what the gate accepts. The quant is never gated. What the gate cannot see: a
view that is swayed by a rumor but cites only an informative filing (an earnings 8-K is in
most windows) still passes; the gate removes the cheap laundering paths, not a model that
hides its reasons.

### 3.15 Re-identification harness (`agents/reid.py`, `aifirm reid-probe`)

Blind mode only helps if the model cannot tell which stock and month a blind sheet is.
`run_reid_probe(llm, factsheets, *, model, role=None, candidate_tickers, month_window,
min_probes=400, max_excess=0.01, max_upper=0.02, max_marginal_excess=0.05,
max_marginal_lower=0.02)` measures it on the exact payload:

* Each real sheet is blinded with `blind_factsheet` and rendered as the full blind sheet
  (default; a superset of every slice) or one analyst's slice (`role`).
* One call per probe (role `reid_probe`) asks for the ticker, chosen from the candidate
  list, and the calendar month of day T, within the month window. Because the attacker
  chooses from a closed list, chance is exact: 1/candidates for the ticker, 1/months for
  the month, and their product for both ("joint").
* Hit rates carry Wilson 95% intervals. A failed call is excluded, not counted as a miss.
* **Gate:** at least 400 answered probes; joint hit rate at most 1 percentage point above
  joint chance; joint upper bound at most 2 points above joint chance; neither marginal
  (ticker, month) more than 5 points above its chance, nor with a Wilson lower bound more
  than 2 points above it; and no payload contains the true ticker, the as-of date or the
  company name verbatim. The marginals matter with a large candidate list: a model that
  dates every sheet but never names the ticker sits near joint chance, yet knowing the
  month is the look-ahead blind mode exists to stop.

`aifirm reid-probe --config F --start D --end D [--n 400] [--role R] [--model M]` samples
(date, ticker) pairs reproducibly (`--seed`) from the rebalance dates and the whole
universe (the candidate list), builds historical sheets (no news), writes a JSON report
and a hash-chained ledger under `<runs_dir>/reid/`, and exits with status 1 when the gate
fails. `--offline` uses the random guesser and needs no credentials. The default model is
`models.committee_model` and the default budget `process.max_llm_cost_usd_per_run`.

Limits: the backtest does not yet refuse a blind pre-cutoff LLM run without a passing
report. Blind mode keeps the SEC SIC sector label, which can nearly name a company on
real data (for example "Electronic Computers").

### 3.16 Books and the quant-only shadow book (`books.py`, `shadow.py`)

A **book** is a directory holding one paper portfolio's state: `ledger.jsonl`,
`ledger_anchors.jsonl`, `forecasts.sqlite`, `lessons.jsonl`, `paper_broker.json`,
`risk_state.json`, `pending_orders.json` and `reports/`. The live book is `<runs_dir>`;
the offline book is `<runs_dir>/offline`. `settle_paper_book` credits each ex-date's
dividend to the shares held at the previous close, then fills orders queued by the previous
decision at the open of the next session in the data.

The **shadow book** of a book lives in `<book>/shadow`. Every `aifirm run` (unless
`--no-shadow` or `evaluation.shadow_book = false`, and only when the live configuration
has an LLM stage) decides it after the live decision is made and queued:

* configuration `quant_only_config(cfg)`: the quant alone, no challenge, memo or risk
  review; the same universe, screen, limits, costs and kill switch;
* the same close, deep copies of the same fact sheets and the same prices;
* `PaperBroker` only, whatever `execution.broker` says;
* on its first run it is forked from the live book's state before the live decision (a
  `shadow_inception` ledger entry); after that it has its own ledger, forecasts, risk state
  and queue, settled by the same code as the live book.

`run_shadow` never raises: a failure is printed and ledgered as a warning and the live
decision stands. Its only write into the live book is one `shadow` forecast per scored
name, which no live decision reads.

**`compare_books`** (`aifirm compare`) reads both ledgers and reports, at the latest close
both books decided:

```
AI value added net of LLM cost = (live NAV − shadow NAV) − LLM spend since inception
```

plus returns since inception, sessions the shadow missed, and the paired per-date rank IC
of the live final probability (`committee`) against `shadow` on the names both forecast,
from forecasts resolved by that date. The verdict is "not yet measurable" until at least
`max(overlay_min_dates, ceil(2 · 252 / step))` resolved dates exist (24 at a 21-day cadence,
504 for daily runs); then "better" if t ≥ 2 with a positive mean difference, "worse" if
t ≤ −2, else "no difference" (default to the simpler design). NAVs come from the ledgers'
`cycle_start` entries; a NAV gap over a short window is mostly noise.

* A session counts as decided only if its cycle completed (`cycle_end`). A cycle that
  started and failed (`cycle_error`, for example an exception inside the decision) is a
  missed session, for the shadow and the live book alike, and the notes say so.
* The NAV at the compared close is the NAV *before* that session's decision (its orders fill
  at the next open), so the LLM spend charged is that of the cycles strictly before it,
  failed attempts included (`cycle_error` logs the spend of the calls made before the
  failure). The spend of the compared session and of every later cycle up to `as_of` (a
  failed latest cycle, whose shadow never ran, included) is shown as pending until fills
  reach a compared NAV, and every failed live session up to `as_of` is listed.
* Dates skipped because fewer than `evaluation.ic_min_names` names had both forecasts are
  counted and reported. The note says the comparison can never become measurable only when
  that is true (`overlay.thin_dates_note`): the screen is on and `deep_dive_top_n` is below
  the minimum, or the universe is smaller than it. Otherwise it says why such a day is
  usually thin (few names with quant P above 0.5 for the deep dive).

### 3.17 Backtest evaluation (`backtest/ic.py`, `random_books.py`, `evaluation.py`, `factors.py`)

Every part is computed after the run from the recorded decisions, independently. A part
that fails is reported as missing with a note, never as a result.

**Clean decisions only (P10).** When the run's models have a training cutoff (llm and
offline modes), the IC, the paired IC tests, the power, the random books and the factor
alpha use only the decisions made after `contamination_end` (and the returns earned under
them). The whole-window IC is kept apart (`Evaluation.contaminated_ic`), shown in a table
labelled "not evidence of skill", and no verdict sentence reads it; with no clean decision
the report says that no clean evidence exists. The cost sensitivity re-executes every
recorded decision and says so.

* **Forward excess return.** Ticker total return minus benchmark total return from the open
  of the execution day (the trading day after the decision) to the close of the
  resolution date, on adjusted prices. No price after the backtest end is used; a window
  ending after it is not evaluated. (Forecasts themselves resolve from the decision close,
  so the two windows differ by the overnight gap.)
* **Signal IC.** Per decision date, the Spearman rank correlation between a signal's
  probabilities and forward excess returns (at least `ic_min_names`, default 5, names).
  Signals: `pool`, labelled "Firm's final probability (traded)" (the overlay or pool
  probability for committee names, the quant's for screened names outside the deep dive), the
  equal-weight variant, each analyst's sealed probability, and the quant-only baseline.
  Reported: mean, std, t-stat, hit rate, names per date. The t-stat is plain when
  `rebalance_every_days ≥ horizon_days`; otherwise Newey-West with `ceil(h/r) − 1` lags.
  Abstentions are not counted as 0.5.
* **Paired IC tests.** Per-date differences: `pool` minus quant-only, and weighted minus
  equal-weight pool.
* **Power.** Decision dates needed for t = 2, `(2s/IC)²`, and for 80% power at 5%
  (two-sided), `(2.80s/IC)²`, where s is the per-date IC std (long-run std when dates
  overlap); years = dates × step / 252; and MinTRL (minimum track record length for 95%
  confidence that the true Sharpe is above zero) in years. The report says "Anecdotal"
  unless the sample already has the dates for 80% power.
* **Composition-matched random books.** 1,000 seeded books (`n_random_books`) that copy the
  strategy's weights, number of names, core, cash and holding periods at every rebalance,
  but hold names drawn at random from the names priced at that decision (optionally from
  the same sector). Every book pays the strategy's realised cost per dollar traded. The
  strategy goes through the same accounting. Reported: percentile and one-sided p-value
  of total return, Sharpe percentile, percentile gross of costs. Below the 95th
  percentile the selection is "not distinguishable from random books". A note warns when
  the universe is too narrow for a strong null.
* **Cost sensitivity and breakeven.** The recorded final targets are re-executed through
  the same engine with every cost component multiplied by k (`cost_multipliers`, default
  0, 1, 2). Re-executing at k = 1 must reproduce the engine's NAV; a mismatch is flagged.
  The breakeven multiplier against the benchmark and against the quant-only baseline is
  found by bisection up to `breakeven_cap` (50). Statuses: `found`, `above_cap`,
  `no_edge`, `gains_above` (the strategy trades less than the comparator). The breakeven
  is also given in basis points per dollar traded, one-way.
* **Cutoff buffer.** `contamination_cutoff = training cutoff + cutoff_buffer_days` (31).
  Decisions on or before it are contaminated. `BacktestResult.model_cutoff` stays the raw
  vendor cutoff.
* **Factor attribution.** OLS of the daily strategy return minus the T-bill return (BIL) on
  the *invested* fraction of NAV at the previous close (`BacktestResult.exposure`), on ETF
  factor proxies with Newey-West errors. The paper broker pays no interest on cash, so
  charging the full T-bill rate would book `(1 − invested) × rf` of cash drag as negative
  alpha (a book 20% in SPY would show about −0.8·rf a year with a huge t-stat); the cash
  drag is reported on its own line instead. Factors: market (SPY − cash), size (IWM − SPY), value (IWD − IWF),
  momentum (MTUM − SPY), quality (QUAL − SPY), low volatility (USMV − SPY). A factor whose
  data cover less than 95% of the days is dropped with a note. Alpha is annualised with
  its t-stat. Proxies are downloaded only when `run_backtest` downloaded its own prices.
* **Six lines** (`report.six_lines`), stated first in every backtest report: window and
  number of assets (and a survivorship flag); model, cutoff plus buffer and where the test
  lies; cost model and breakeven; configurations tried; deflated Sharpe at that count
  (PSR when only one trial exists); benchmark and the random-book null.

### 3.18 Defaults and their research source

These are engineering defaults from the research, not optimised values.

| Setting | Default | Source |
|---|---|---|
| `portfolio.max_position` | 0.08 (sizing target) | RESEARCH 6.4: "≤ 8% target" |
| `risk.max_position_hard` | 0.10 | 6.4: "10% hard cap" |
| `portfolio.max_sector` | 0.25 | 6.4: "≤ 25% absolute" |
| `portfolio.target_vol` | 0.15 (excess in cash) | 6.4: 12–15% annualised |
| `portfolio.no_trade_band` | 0.005 | 6.4: 0.5%-of-equity band |
| `portfolio.max_turnover` | 0.25 one-way | 6.4: ≤ 25% one-way per rebalance |
| `portfolio.rebalance_fraction` | 0.5 | 6.4: move 50% of the way; 6.5: partial rebalancing cut turnover |
| `portfolio.kelly_fraction` | 0.25 | 6.3: at most a quarter to a half Kelly |
| `risk.drawdown_derisk` | 0.10, re-risk after half recovery | 6.4 |
| `risk.drawdown_freeze` | 0.20, human reset | 6.4 |
| `risk.max_adv_participation` | 0.02 | stricter than 6.4's 5% |
| `process.combination` | `overlay` | 6.6 |
| `process.overlay_max_tilt` | 0.05 | 6.6 asks for a bound; the value is an engineering choice |
| `process.overlay_min_dates` | 12 | engineering choice |
| `process.believability_min_dates` | 20 | 6.2: "after at least about 20 graded dates" |
| `calibration.min_n`, `min_dates`, `prior_strength` | 30, 20, 50 | 4 and 6.1 (overconfidence; calibrate); 6.2 item 3 (about 20 graded dates); values are engineering choices |
| `universe.screen.deep_dive_top_n` | 12 (screen off by default) | 9.2 and 9.3 item 1 |
| `evaluation.n_random_books` | 1000 | 9.3 item 6 |
| `evaluation.cost_multipliers` | 0, 1, 2 | 7.4 S0: results must survive 2× costs |
| `evaluation.cutoff_buffer_days` | 31 | 7.2: start the clean window a month after the cutoff |
| `evaluation.shadow_book` | true | 1, 6.6 and 9.3 item 6 |
| reid gate | 400 probes, +1 pt, upper +2 pt | 7.2 |
| `process.news_reader_enabled`, `tier_gate_enabled` | true | 5.5 and 9.3 item 3 |

## 4. What is deliberately *not* automated

* Moving from paper to real money, raising capital limits and resetting a frozen book are
  **human decisions**. `aifirm unfreeze --reason "..."` is the reset: it asks for
  confirmation, restarts the high-water mark at the NAV of the book's last decision and
  writes the before/after state and the reason to the ledger.
* Orders to Alpaca need a human approval at the prompt, unless `--yes` is passed or
  `execution.require_approval` is set to false. Use `--yes` only for a supervised schedule.
* Managing other people's money brings legal obligations (adviser registration, rules on
  performance claims). This software is a research tool, not investment advice.

## 5. Known gaps against the research

The research recommends several controls the code does not implement yet:

* **Risk:** no ex-ante beta band, no sleeve or per-agent drawdown limits, no price-shock
  fast path, no stop-triggered re-review (RESEARCH 6.4, 5.5). Drawdown thresholds are fixed,
  not scaled to expected volatility.
* **News:** tier B needs one established outlet, not two independent ones; no tier-C hold;
  no historical (point-in-time) news archive, so news is never backtested. The close-time
  calendar knows the scheduled half days (13:00) but no ad-hoc closures. In a live run the
  headline feed is fetched for every universe name, although only the deep-dive names'
  headlines are read.
* **Aggregation:** no per-agent cross-sectional de-biasing (median/MAD standardisation,
  RESEARCH 6.2 item 1). The overlay adds the LLM pool's own edge (`p_llm − 0.5`) to the
  quant, so an LLM pool with a persistent optimistic bias tilts every name up until
  calibration (after 30 resolved questions from 20 dates per agent) removes the bias, and
  an LLM that agrees with the quant adds to it (bounded by `max_tilt` and the adaptive
  bound). No extremising; no market-implied prior. Believability does not reset when an
  analyst's model changes (calibration does).
* **Contamination:** cutoffs are vendor dates plus a buffer, not measured cutoffs (the
  forecast store's `pre_cutoff` flag and the backtest use the same buffer). Prompt edits do
  not change the config fingerprint; the trial ledger counts them through a digest of the
  prompt modules' source (`backtest.engine.prompt_digest`), any edit to those modules
  included. The backtest does not refuse a pre-cutoff LLM run without blind mode and a
  passing re-identification report (its skill evidence uses only clean decisions, and its
  contaminated returns are reported apart). Blind mode keeps the SEC SIC sector label,
  which can nearly name a company on real data.
* **Evaluation:** no probability of backtest overfitting (PBO), no vintage ladder across
  several models, no one-use holdout enforcement, no regime breakdown
  (see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)).
* **Data:** Yahoo prices omit most delisted names, and the bundled universe list is
  today's membership (survivorship bias).
