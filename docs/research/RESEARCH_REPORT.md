# Building an AI-run investment firm: what the research says

*Research report for the `aifirm` project, compiled 17 November 2026. It covers published work through
mid-November 2026. This is not investment, legal or tax advice.*

---

## How this report was made, and how to read it

The report draws on 14 research lanes. They covered academic multi-agent trading frameworks, a
code-level read of the two largest open-source projects (TradingAgents and ai-hedge-fund), the
wider open-source ecosystem, how real investment firms organise decisions, known failure modes,
the data layer, live benchmarks, portfolio construction, legal and operational matters, and four
"gap" lanes. The gap lanes covered LLMs as feature extractors, how to test when models have
memorised history, adversarial attacks on data-gathering agents, and contradictions between
sources. A separate fact-check pass corrected several widely repeated numbers. **Where a
correction exists, this report uses the corrected version.**

Limits you should know about:

- Some hosts could not be reached from the research sandbox, among them nof1.ai (Alpha Arena),
  SSRN, NBER, several journal sites and some vendor pricing pages. The shared web-search budget
  ran out partway through. A few figures therefore rest on secondary sources or recollection, and
  they are labelled as such.
- Many of the most relevant papers are 2026 preprints. Several are by a single author and have
  not been peer reviewed. They are cited because they are the best available evidence, not
  because they are settled.

**Confidence tags** used below:

- **[H]**: checked against the primary paper, code or official documentation.
- **[M]**: credible secondary source, or a number from a paper whose full text could not be re-read.
- **[L]**: recollection or press report only. Do not build on it without checking.

**Short glossary.**

- **Alpha**: return beyond what market and style exposure explain.
- **Beta**: exposure to the overall market.
- **Sharpe ratio**: return per unit of volatility. Above 1 a year sustained over many years is
  very good; above 3 is almost never real.
- **IC (information coefficient)**: correlation between forecasts and later returns across
  stocks. An IC of 0.02 to 0.05 is typical of a genuinely useful signal.
- **Point-in-time (PIT)**: using only data that was actually available on the decision date.
- **Look-ahead bias**: accidentally using later data.
- **Survivorship bias**: testing only on companies that still exist today.
- **Training cutoff**: the date after which a language model has seen no data.
- **Brier score**: squared error of a probability forecast. Lower is better.

---

## 1. Executive summary

### The honest bottom line

As of November 2026 there is **no audited, multi-year, after-cost evidence, measured after the
models' training cutoffs, that LLM agents produce durable stock-selection alpha**. The striking
results that made multi-agent trading famous do not survive honest re-testing:

- TradingAgents reported a Sharpe of 8.21 on Apple, but over three months and three tickers. [H]
- HedgeAgents reported +400% over three years, mostly in a period before its model's cutoff. [M]
- FinCon reported +114%, over eight months inside GPT-4-Turbo's training period. [H]

What happens under careful testing:

- **Long horizons erase the edge.** FINSABER re-ran two leading LLM agents over 2004–2024 on a
  survivorship-free S&P 500 universe. Buy-and-hold beat both on risk-adjusted terms, and neither
  had statistically significant alpha (all p > 0.34). The agents were too cautious in bull
  markets and too aggressive in bear markets. [H]
- **Moving past the training cutoff erases the edge.** Re-testing five published agents after
  GPT-4o's cutoff cut their Sharpe ratios by 51–62%. [H]
- **The best-controlled 2026 benchmark found beta, not alpha.** It ran 91 runs of 6 models with
  minute-stamped news and no look-ahead. It found no significant alpha. Market and style exposure
  explained a median 78% of returns, and agents earned as much with their headlines removed. [H]

Live contests point the same way:

- In Alpha Arena Season 1, four of six frontier models lost real money. [L]
- In LiveTradeBench, every one of 21 models trailed an equal-weight buy-and-hold of the same 15
  stocks over the same 50 days. That figure is the fact-checker's calculation; the paper itself
  reports no passive baseline. [M]

The most credible positive results are modest and narrow:

- **MarketSenseAI.** Its live-archived monthly signals beat equal weight on an S&P 500 cohort by
  +25.2 percentage points over 19 months (p = 0.003). No individual agent's date-level IC was
  significant. [H]
- **Chen & Pu.** An agentic web-search model's top-20 picks earned 18.4 bps/day of factor alpha
  (t = 2.46), but over only about nine months and with one model. [M]
- **Lopez-Lira & Tang.** GPT-4 headline sentiment produced a gross Sharpe of 2.97, but the
  strategy is unprofitable at 20 bps round-trip costs, and its Sharpe fell from 6.54 to 1.22
  between 2021 and 2024 as adoption spread. [H]

### What works

1. **Organising the firm into specialist roles with structured outputs.** This is a good
   engineering pattern for coverage and auditability. It is not, by itself, a source of returns.
2. **Deterministic risk and sizing outside the LLM.** This is the strongest design signal in the
   literature.
   - FinCon: 113.8% with a CVaR risk trigger vs 14.7% without, in-sample. [H]
   - FARSIGHT: in simulation, a rule-based price-shock freeze cut a fake-headline crash from 9.6%
     to 1.1%. [H]
   - Barton et al.: LLM sizing in live production fleets ignored volatility entirely. [H]
3. **Point-in-time data plus honest evaluation.** Even the flagship open-source framework
   shipped about ten releases of look-ahead fixes in 2026. [H]
4. **Independent views combined statistically.** Majority voting explains most of the gains
   credited to multi-agent debate (NeurIPS 2025). [H] Calibrated probability forecasts anchored
   on market prices reach superforecaster level in forecasting tournaments. [H]
5. **LLMs as measurement tools feeding a quant core, rather than as traders.** On identical news
   data, embeddings plus a trained model gave Sharpe ratios of about 4.8–4.9. Prompting small
   models for sentiment on the same data gave 0.95–1.76 (gross, daily, equal-weighted). [H]

### What does not work

- Letting an LLM choose position sizes, leverage or market timing.
- Famous-investor personas.
- Treating bull/bear debate as a source of alpha.
- Trusting short-window backtests.
- Treating LLM-discovered trading rules as discoveries before counting how many were tried. An
  honest-evaluation study rejected every one. [H]

### What this means for the build

Build a firm where **LLMs propose and code disposes**:

- Analysts produce sealed, calibrated, evidence-cited probability forecasts from different slices
  of point-in-time data.
- Forecasts are pooled statistically, with argument only on disagreement.
- Deterministic code sizes positions and enforces an invariant risk core.
- Everything is logged and scored.

Run the LLM layer as a bounded overlay on a quant/passive core. Keep a quant-only shadow book so
you can measure whether the LLMs add anything. Count only forward, post-cutoff, after-cost
evidence. Expect the honest answer to take years for a narrow book and months only for a broad
one (Section 7.3).

---

## 2. How others have built it

### 2.1 Comparison of major frameworks and projects

| Project (year) | Roles | Decision protocol | Data | Evaluation | Reported result | Key caveats |
|---|---|---|---|---|---|---|
| **TradingAgents** (Tauric/UCLA/MIT, 2024–26) [paper](https://arxiv.org/abs/2412.20138), [repo](https://github.com/TauricResearch/TradingAgents) | 4 analysts, bull & bear researchers, research manager, trader, 3 risk debaters, portfolio manager | LangGraph pipeline. Debates are judged by LLM managers and produce a 5-tier rating. Risk is LLM debate only. | yfinance, Alpha Vantage, SEC EDGAR as filed, FRED vintages, news, social | Paper: Jan 1–Mar 29, 2024; AAPL, GOOGL, AMZN | AAPL CR 26.62%, Sharpe 8.21 [H] | 3 months, 3 tickers, before the model cutoff. "Alpha Illusion" reproduction: net of costs it trailed buy-and-hold on 4 of 5 tickers. Many look-ahead leaks fixed in 2026. |
| **FinCon** (NeurIPS 2024) [paper](https://proceedings.neurips.cc/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html) | Manager + analysts, each specialised in one data source | Analysts report only to the manager. CVaR risk alert. Between-episode "belief" updates. | News, filings, calls, quant metrics | Test Oct 2022–Jun 2023, GPT-4-Turbo | 3-stock portfolio 113.8% (Sharpe 3.27) with CVaR vs 14.7% without [H] | Inside the training period. Largest post-cutoff Sharpe decay (62%) in Profit Mirage. |
| **FinMem** (2023) [paper](https://arxiv.org/abs/2311.13743) | Single agent, layered memory, persona | Memory retrieval, then decision | News, filings, prices | Single tickers, short windows | TSLA about +62% [M] | MSFT +23.3% became −22.0% when the FinCon paper reported FinMem over a window about 2 months longer. FINSABER: no alpha. |
| **FinAgent** (KDD 2024) [paper](https://arxiv.org/abs/2402.18485) | Multimodal single agent with two-level reflection | Tool-augmented decision | Prices, news, chart images | 6 datasets | About +36% average profit improvement [M] | FINSABER: Sharpe 0.24 vs buy-and-hold 0.70 [H] |
| **HedgeAgents / QuantAgents** (2025) [1](https://arxiv.org/abs/2502.13165), [2](https://arxiv.org/abs/2510.04643) | Manager + asset experts holding "meetings" | Budget, experience-sharing and extreme-market meetings | Multi-asset | Multi-year, mostly before the cutoff | about +400% / about +300% over 3 years [M] | Implausible magnitudes. Unreplicated. |
| **AlphaAgents** (BlackRock, 2025) [paper](https://arxiv.org/abs/2508.11152) | Fundamental, sentiment, valuation | Debate to consensus, then equal weight | Fundamentals, news | 15 tech stocks, Feb–May 2024 | Beat benchmark (risk-neutral setting) [M] | 4 months, before the cutoff. Authors say it is not an optimiser. |
| **ContestTrade** (2025) [paper](https://arxiv.org/abs/2508.00554) | Data team + research team with configurable "beliefs" | Internal contests. Realised scores feed a LightGBM forecast of each agent's Sharpe; capital goes to agents with positive predicted Sharpe. | A-share news, announcements, flows | Jan–Jun 2025, 0.1% costs, T+1, single run | Sharpe 3.12, CR 52.8% [H] | One 6-month run. Removing one contest gives Sharpe 1.78 or 2.01. "All removed" (0.07) also removes the research tools. |
| **MarketSenseAI** (2024–26) [2.0](https://arxiv.org/abs/2502.00415), [2026](https://arxiv.org/abs/2604.17327) | News, fundamentals, price dynamics, macro, synthesis | LLM synthesis, monthly buy/hold/sell | Filings (retrieval), news, macro | 2026 paper: signals archived live | S&P 500 cohort +25.2 pp over 19 months (p = 0.003). S&P 100 +30.5% over 35 months (p = 0.17). [H] | Agent-level date IC not significant. Vendor-authored. |
| **Expert Investment Teams** (Oxford, 2026) [paper](https://arxiv.org/abs/2602.23330) | Fine-grained task agents (technical, quant, qualitative, news, sector, macro, PM) | Scores feed a monthly long-short | TOPIX 100 | Sep 2023–Nov 2025, leakage-controlled | Fine-grained tasks beat coarse roles on Sharpe [H] | Agents alone did not beat the index's Sharpe (secondary figures [M]). Which agent mattered depended on prompt design. |
| **ai-hedge-fund v1** (virattt) [repo, last v1](https://github.com/virattt/ai-hedge-fund/tree/6c41ae8cf5fb4e30eb8cdb6a816eb56ff234989d/src) | 13 investor personas + 6 analysts, deterministic risk manager, LLM PM | Parallel fan-out. PM picks a quantity within shown limits. | Financial Datasets | Daily backtester | None claimed | Same-day fills, zero costs, fundamentals leaked by report date [H]. The LLM's quantity is not re-clamped in code. |
| **ai-hedge-fund v2** (Nov 2026) [repo](https://github.com/virattt/ai-hedge-fund) | LLM persona models + a post-earnings-drift quant model, all emitting Signals in [−1, 1] | Deterministic blend, 7 shrink-only risk clamps, analyst weights set by IC t-stat | Filing-date-filtered fundamentals | One engine for backtest and paper, blind mode | None claimed | Validation module (CPCV/PBO) is an empty stub. Live broker not built. |
| **Vibe-Trading** (HKUDS, 2026) [repo](https://github.com/HKUDS/Vibe-Trading) | YAML-defined committees (bull, bear, risk, PM) | LLM committee, then a fail-closed live order gate | 28 sources with fallback | Engineering checks | None claimed | Committee value unmeasured. Best execution-safety design found. |
| **FinRobot V2** [repo](https://github.com/AI4Finance-Foundation/FinRobot) | Data, analysis, modelling, synthesis, report; bull/bear/judge | "Models reason, software computes" | 7 providers + cross-provider validator | Research reports | None claimed | A research tool, not a trader. |
| **R&D-Agent(Q)** (Microsoft, NeurIPS 2025) [paper](https://arxiv.org/abs/2505.15155) | LLM researcher + coder, qlib backtester | Bandit chooses factor vs model research | China A-shares (qlib) | Test 2017–2020; v2 adds 2024–25 out-of-sample | IC 0.0532, annual return 14.21%, IR 1.74, for under $10 of LLM cost [H] | Main test is before the cutoff. Its LightGBM baseline (IR 0.57) is far weaker than qlib's own (IR 1.02). |
| **DeepFund** (NeurIPS 2025 D&B) [paper](https://arxiv.org/abs/2505.11065) | Planner, analysts, PM | LLM proposes a position ratio; code clamps it to 2/N | Alpha Vantage, yfinance | Live, Mar 17–Apr 17, 2025 | 8 of 9 models lost money in the main window [H] | 5 stocks, weeks long. In the Q2 extension, 6 of 9 ended positive. |
| **Trading-R1 / FLAG-Trader / TradingGroup** (2025) [Trading-R1](https://arxiv.org/abs/2509.11420) | Fine-tuned or RL-trained trader models | Trained policy | 14 equities (Trading-R1) | About 3 months, 6 tickers | NVDA Sharpe 2.72 (self-reported) [M] | Short, unreplicated, no portfolio. |
| **FinPanel** (ICAIF 2026 workshop) [paper](https://arxiv.org/abs/2611.18150) | Per-stock recruited trader-persona panel + manager | Manager weighs arguments against track records | FINSABER data | 8 US stocks, 2025 | 16.1% vs 7.9% for the best prior agent [H] | No LLM method beat buy-and-hold (21.3%). Track records changed nothing (16.1 / 15.6 / 16.2%). |
| **Bridgewater AIA Forecaster** (2025) [paper](https://arxiv.org/abs/2511.07678) | About 10 independent search agents + supervisor | Supervisor reconciles; statistical (Platt) calibration; blend with market | News search | ForecastBench and prediction markets | About superforecaster level. Ensemble with the market beat the market alone. [H] | A forecasting system, not a trading system. |

### 2.2 Notes on each

**TradingAgents** is the reference "virtual trading firm" and the most widely copied design. Its
paper's results are not evidence:

- Table 1 covers January 1 to March 29, 2024, on AAPL, GOOGL and AMZN. Version 1 of the paper
  also mentions a June–November window, which is an internal error. [H]
- The authors themselves say the Sharpe "exceeds our expected empirical range". [H]
- The buy-and-hold baselines it prints (AAPL −5.23%) cannot be reproduced from Yahoo data for
  that window (about −7.6%). [H]
- The ["Alpha Illusion"](https://arxiv.org/abs/2605.16895) reproduction ran one year (2025) on
  TSLA, NVDA, KO, XOM and MSTR. TradingAgents ended at $106.4k gross and $102.3k net, against
  $104.8k for buy-and-hold, which beat it net of costs on 4 of 5 tickers. [H]

The repository is worth studying for its engineering. In 2026 it fixed a series of real
look-ahead leaks (CHANGELOG [H]):

- FRED macro data served from today's revised figures;
- undated Reddit and StockTwits posts;
- memory returning lessons whose outcomes were not yet known on the trade date;
- Alpha Vantage fundamentals leaking future values;
- unpadded date strings ("2024-7-5") comparing as later than real dates and leaking filings.

It also fixed hallucination bugs:

- the wrong company being analysed;
- invented indicator values;
- a debater rebutting a fabricated, empty opponent;
- a quoted "Consensus rating: Buy" overriding the portfolio manager's own rating.

Its risk team is entirely LLM debate, with no deterministic limit anywhere in the code. Its
built-in backtester scores decisions only (hit rate and 5-day alpha). It has no portfolio NAV,
costs or Sharpe. Prices come with `auto_adjust=True`, so historical price levels reflect later
splits and dividends. [H]

Running it costs about 16–31 LLM calls per ticker per day at debate depth 1 and 36–51 at depth 5.
These figures are derived from the code, not measured. [M]

**ai-hedge-fund** shows the field's direction better than any paper. Version 1 had LLM personas
and an LLM portfolio manager. Its backtester filled orders at the same day's close with zero
costs and used fundamentals dated by fiscal period end rather than filing date, which is
look-ahead. [H]

Version 2 (the default since August 2026) is rebuilt on the rule "the LLM never touches the
trade" [H]:

- Personas and a quant model emit signals.
- Deterministic code blends them and applies risk clamps that can only shrink the book: per-name,
  net, sector, beta, volatility, drawdown halving and gross.
- Costs are charged: half-spread, commission and square-root market impact.
- Decisions are made at the close of day T and executed at the next close.
- A hash-chained ledger records everything.
- Analyst weights move with their IC t-statistic, with a floor and a cap.
- Backtests run personas "blind", without ticker, dates or dollar values.

Two caveats. The code itself concedes that blind mode "reduces [recall] rather than removing it".
Its overfitting checks (CPCV/PBO) are not yet written. [H]

**FinCon, FinMem and FinAgent** are the 2023–24 generation. Each tested on a few stocks for a few
months inside the model's training period. FINSABER's long-horizon re-test
([paper](https://arxiv.org/abs/2505.07078), KDD 2026 Datasets & Benchmarks oral) is the
definitive check:

- Universe: 63–91 point-in-time S&P 500 names per selection scheme, 2004–2024.
- In the Volatility-Effect selection, Sharpe was 0.703 for buy-and-hold, 0.241 for FinAgent and
  −0.228 for FinMem. [H]
- By market regime, Sharpe in bull and bear years was:
  - FinAgent: 0.12 and −0.38
  - FinMem: −0.19 and −0.97
  - buy-and-hold: 0.61 and −0.28 [H]
- On raw annual return the LLM agents occasionally led (FinAgent 13.95% under Momentum
  selection). The failure is in risk-adjusted terms. [H]
- One frequently quoted number needs correcting. FinMem's MSFT result falling from +23.3% to
  −22.0% is not a FINSABER re-run. It compares FinMem's own paper with FinMem as reported in the
  FinCon paper over a window about two months longer. [H]

FinCon's CVaR ablation is still the most-cited sign that a quantitative risk trigger matters: a
113.8% return with it vs 14.7% without. It is in-sample on three stocks. [H]

**ContestTrade** is the most interesting idea for aggregation: score every agent on later
outcomes and give capital only to agents with good predicted risk-adjusted performance. The
headline ablation is widely misread.

- Removing everything collapses Sharpe from 3.12 to 0.07, but that version also removes the
  deep-research tools (and, per v5, the LLM judge).
- Removing only the researcher contest gives 1.78. Removing only the data contest gives 2.01. [H]
- It is a single six-month A-share backtest. The authors call the evidence preliminary.
- Its contests' rank ICs were modest: 0.054 and 0.079. [H]

**MarketSenseAI** is the best example of honest live evaluation: signals were generated and
archived at each date, and attribution is reported per agent. Its results are mixed but real.
The strong-buy portfolio on an S&P 500 cohort earned +2.18% a month vs +1.15% for equal weight
(p = 0.003). Pooled IC was about +0.05 for fundamentals, negative for price dynamics, and about
zero for news. [H]

**Expert Investment Teams** (TOPIX 100, monthly) found that precisely specified analyst tasks beat
generic roles. It also found that which agent mattered depended on prompt granularity: the
technical agent in fine-grained prompts, news in coarse ones. [H] The lesson is to measure agent
contributions by ablation, not assume them.

**FinRobot V2 and Vibe-Trading** contribute engineering patterns rather than performance claims:

- FinRobot computes every valuation number in deterministic operators. It audits LLM prose
  against the numbers and cross-checks two data providers (revenue within 15%, market cap
  within 5%). [H]
- Vibe-Trading's live order guard fails closed through a fixed sequence of checks: mandate,
  expiry, kill switch, account binding, positions read from the broker, then hard caps on
  notional, exposure, leverage and trades per day. [H]

**R&D-Agent(Q) and AlphaAgent** use the LLM as a quant researcher. It proposes factor and model
code, and a deterministic backtester judges it. This is cheap (under $10 per run) and productive,
but it is a multiple-testing machine. In the "What survives honest evaluation" study, every
LLM-discovered strategy failed once its own trial count was taken into account
([2608.27734](https://arxiv.org/abs/2608.27734)) [H]. One strategy with a Sharpe near 3 in
development scored 0.26 on its single-use holdout ([Hard to Fool](https://arxiv.org/abs/2611.06607)) [H].

### 2.3 Live and forward benchmarks

| Benchmark | Setup | Main finding | Caveats |
|---|---|---|---|
| **Nof1 Alpha Arena S1** ([site](https://nof1.ai)) | 6 frontier LLMs, $10k real money each, crypto perpetuals on Hyperliquid, about Oct 17/18–Nov 3, 2025 | 4 of 6 lost money. Reported finals: Qwen3 Max ≈ +22%, DeepSeek ≈ +5%, Claude Sonnet 4.5 ≈ −31%, Grok 4 ≈ −45%, Gemini 2.5 Pro ≈ −57%, GPT-5 ≈ −63% (aggregate ≈ −28%) [L] | One run per model, about 17 days, leverage chosen by the LLM. Behavioural explanations unverified. Season 2 status unknown. |
| **LiveTradeBench** ([paper](https://arxiv.org/abs/2511.03628)) | 21 LLMs, 50 live days (Aug 18–Oct 24, 2025), 15 US large caps + Polymarket | Stock returns +1.78% to +6.25%. Equal-weight buy-and-hold of the same stocks returned about +7.5% (fact-checker calculation) [M]. General benchmark rank (LMArena) vs return: Spearman 0.054 on stocks, −0.38 on Polymarket [H] | No costs, single runs. Sharpe across the two markets is uncorrelated. |
| **AI-Trader** ([paper](https://arxiv.org/abs/2512.10971)) | 6 LLMs, Nasdaq-100 hourly, Oct 1–Nov 7, 2025 | MiniMax-M2 +9.56%, DeepSeek-v3.1 +8.39%, Claude-3.7 +3.11% vs QQQ +1.87%. No agent beat SSE-50 in A-shares [H] | About 5 weeks. The README's Oct 31 snapshot shows different numbers. The "SR" column is Sortino. |
| **StockBench** ([paper](https://arxiv.org/abs/2510.02209)) | 20 DJIA stocks, Mar–Jun 2025 | Most models struggle to beat buy-and-hold (+0.4%) [M] | 4 months; the paper's text is internally inconsistent. |
| **DeepFund** ([paper](https://arxiv.org/abs/2505.11065)) | 9 LLMs, live, 24 trading days in Mar–Apr 2025 | 8 of 9 lost money [H] | 5 stocks; covers the April 2025 tariff shock. |
| **Agent Market Arena** ([paper](https://arxiv.org/abs/2510.11695)) | Agent frameworks × LLM backbones, live | Framework drives behaviour more than the backbone does [M] | Few assets, short window. |
| **rta-bench-1 / "Earn Beta, Not Alpha"** ([paper](https://arxiv.org/abs/2611.19530)) | 91 runs, 6 models, 56 US large caps, 3 regimes, news released by publication minute | No significant alpha (largest t = 1.79). Median 78% of return explained by market and style. No change when headlines are removed. A simple trend rule beat every agent [H] | Single-author preprint. Low power: alpha standard error about 7% a year. |
| **ForecastBench** ([data](https://github.com/forecastingresearch/forecastbench-datasets)) | Contamination-free forecasting questions | Best AI ensembles 69.4 vs superforecasters 68.1 (not significant). Best off-the-shelf model 61.0 vs superforecasters 66.9. On market questions, copying the prior-day market price scores 70.2 [H] | Not trading. Shows that the market price is a very strong prior. |

### 2.4 The 2025–26 critique wave

- **Profit Mirage** ([paper](https://arxiv.org/abs/2510.07920)). Five agents on GPT-4o (cutoff
  October 2023), NASDAQ-100, Q2–Q3 2021 vs Q3–Q4 2024. Sharpe decayed by 51.5% (QuantAgent) to
  62.2% (FinCon); total return by 50.2% to 71.9%. [H]
- **Pre-registered audit of 91 LLM trading papers**
  ([2611.18934](https://arxiv.org/abs/2611.18934)):
  - 51% deduct no trading costs, and only 9% model costs from data;
  - the median test window is 12 months, and a quarter are 6 months or shorter;
  - of the 51 papers whose window may overlap training data, 98% have no leakage control;
  - only 5 of 75 combine costs, a leakage control and a reported trial count. [H]
- **Xia et al.** ([2605.19337](https://arxiv.org/abs/2605.19337)). Version 2 (October 2026)
  covers 80 primary studies. Coverage has improved: 62.5% use time-consistent splits and 58.8%
  model costs. But none reaches the top reproducibility tier, and 45 of 80 are at the lowest. [H]
- **"What survives honest evaluation"** ([2608.27734](https://arxiv.org/abs/2608.27734)). A
  deliberately leaky oracle with a Sharpe of 35 passes both standard overfitting tests (deflated
  Sharpe and PBO). Leakage must therefore be blocked structurally, not caught statistically. [H]
- **Production fleets** ([Barton et al., 2609.05663](https://arxiv.org/abs/2609.05663)). Six
  months of live LLM agents trading crypto: 3,505 vaults and 7.5 million model calls.
  - Median leverage was 5.0× in every volatility bucket.
  - The operator's risk slider explained leverage better than the strategy text did.
  - A mechanical 2%/4% stop/target bracket added +39 bps per position, mostly by avoiding
    liquidations.
  - There was no directional edge. [H]

---

## 3. Lessons from real investment firms and multi-agent research

### 3.1 What real firms do

**Separate functions and give each a hard interface.**

- Research produces views.
- A portfolio manager turns views into positions.
- An independent risk function sets and enforces budgets.
- Execution is measured against the price at the moment of decision. This is Perold's
  *implementation shortfall*: the gap between a "paper portfolio" traded at decision prices and
  the real one.
- Compliance runs pre-trade controls in code. SEC Rule 15c3-5 requires automated pre-trade limits
  for market access. Knight Capital lost about $440–460M in roughly 45 minutes in 2012 from a
  deployment error and was the rule's first enforcement case
  ([Chicago Fed](https://www.chicagofed.org/-/media/publications/chicago-fed-letter/2012/cfloctober2012-303-pdf.pdf)). [M]

**Multi-manager "pod" platforms** take this furthest:

- Many small books each get a risk budget.
- A central team neutralises unwanted market and factor exposure. Citadel's risk group describes
  this ([Risk.net](https://www.risk.net/awards/7955800/hedge-fund-of-the-year-citadel)). [M]
- Losing books are cut in stages. Press reports say a Millennium PM's capital is halved at a 5%
  loss and the PM is usually removed at 7.5%. The firm does not disclose this
  ([via AOL/Business Insider](https://www.aol.com/strategy-69-billion-hedge-fund-193733084.html)). [L]

Their known weakness is that apparently independent pods crowd into the same trades. BlackRock
warned about this in spring 2026 [L]. Agents built on one or two foundation models will share
this weakness even more strongly.

**Form views independently, then combine them statistically.** This is the best-supported finding
in group judgment:

- Seeing others' estimates shrinks diversity without improving accuracy
  ([Lorenz et al. 2011, PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC3107299)). [H]
- Groups discuss mostly what everyone already knows and bury facts held by one member. This is
  the "hidden profile" effect (Stasser & Titus 1985). [M]
- Debating in small groups of five, then averaging as few as four group answers, beat a crowd of
  thousands
  ([Navajas et al. 2018](https://ideas.repec.org/a/nat/nathum/v2y2018i2d10.1038_s41562-017-0273-4.html)). [H]
- In the Good Judgment Project, team forecasts beat prediction markets only after statistical
  aggregation: time decay, track-record weighting and recalibration
  ([Atanasov et al. 2017](https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2374)). [M]
- Analysts trained in a structured reasoning technique (Analysis of Competing Hypotheses) were not
  more accurate. Statistical recalibration and aggregation helped substantially
  ([Mandel et al. 2018](https://www.cambridge.org/core/journals/judgment-and-decision-making/article/boosting-intelligence-analysts-judgment-accuracy-what-works-what-fails/1530E9DAE8F42B2E8FC2B5FE374DB50F)). [M]
- Noise reduction is the largest single component of these accuracy gains
  ([Satopää et al. 2021](https://ideas.repec.org/a/inm/ormnsc/v67y2021i12p7599-7618.html)). [M]
- Kahneman, Lovallo and Sibony's
  [Mediating Assessments Protocol](https://sloanreview.mit.edu/?p=65293) applies the same idea in
  practice: independent component assessments, estimate–talk–estimate, then the overall call.
- Bernd Scherer applies it to investment committees
  ([CFA Institute Research Foundation, 2026](https://rpc.cfainstitute.org/research/foundation/2026/investment-committees-governance-design-choices)).
  Members submit portfolios anonymously, each scaled to the same risk, and the committee
  averages them. His 2026 evidence comes from an LLM-simulated committee. [L]

**Weight opinions by measured track record, with care.**

- Bridgewater weights votes by "believability" (at least three successes plus a good explanation)
  and shows the equal-weighted and weighted results side by side
  ([Principles](https://www.principles.com/principles/83b7a067-cd3b-4e57-aaf0-8379ac81b1cb/)). [M]
- Good Judgment used track-record-weighted aggregators. [M]
- Track records are noisy, though. A 2024–25 reanalysis questions how much the training and
  teaming effects were worth. [L]

**Make dissent authentic.** A minority that genuinely disagrees improves group solutions more than
a role-played devil's advocate
([Nemeth et al. 2001](https://zendy.io/title/10.1002/ejsp.58)) [M]. The *pre-mortem* means
imagining the decision has already failed and listing why. It reduced overconfidence more than
other critique methods in one controlled study (Veinott et al. 2010;
[Klein 2007](https://hbr.org/2007/09/performing-a-project-premortem)). The evidence base is
thin. [M]

**Separate idea generation, sizing and selling.**

- Managers' highest-conviction "best ideas" outperform their other holdings by about 1–2.5% a
  quarter ([Cohen, Polk & Silli](https://personal.lse.ac.uk/polk/research/BestIdeas.pdf), a
  working paper). [M]
- Institutional managers' sales underperform random selling, while their buys show skill
  ([Akepanidtaworn et al., JF 2023](https://nber.org/papers/w29076)). [M]
- Vendor data (Essentia, Alpha Theory) says managers add more through picking than sizing, and
  that equal weighting often beats their own sizing. These are client samples, not peer
  reviewed. [L]
- Portfolio constraints pass only a fraction of forecasting skill into positions (the "transfer
  coefficient", [Clarke, de Silva & Thorley 2002](https://ideas.repec.org/a/taf/ufajxx/v58y2002i5p48-66.html)). [M]

**Use different decision rules for different decisions.** In venture capital, a "champion" rule
lets one partner's conviction fund a skewed bet. Majority rules trim outliers
([Malenko et al., JF](https://ideas.repec.org/p/hbs/wpaper/21-131.html)) [M].

**Keep process artifacts.** These include decision journals, checklists built from past mistakes,
base rates from reference classes, mistake logs and investment-committee memos with written kill
criteria. In a McKinsey survey of 1,048 decisions, process quality explained about six times
more outcome variance than depth of analysis
([Lovallo & Sibony 2010](https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-case-for-behavioral-strategy)).
The result is correlational and self-reported. [M]

**How leading firms actually use AI (2024–26).**

- Man Group's "AlphaGPT" proposes, codes and backtests signals. Survivors go through the same
  investment committee as human research
  ([Bloomberg](https://www.bloomberg.com/news/articles/2025-07-10/man-group-says-agentic-ai-is-now-devising-quant-trading-signals)). [M]
- Balyasny has a central AI team building agents that pods tailor
  ([OpenAI case study](https://openai.com/index/balyasny-asset-management)). [M]
- Bridgewater's AIA Labs blends calibrated LLM forecasts with market consensus and keeps risk and
  execution with humans. [M]

None of them has disclosed an autonomous LLM book with audited excess returns. Systematic and
discretionary hedge funds perform about the same once risk factors are accounted for
([Harvey et al. 2017](https://papers.ssrn.com/abstract=2880641)). [M]

### 3.2 What multi-agent LLM research says

**Debate is mostly voting in disguise.**

- [Choi, Zhu & Li](https://arxiv.org/abs/2508.17536) (NeurIPS 2025) separated multi-agent debate
  into voting and argument across 7 benchmarks. Majority voting explains most of the gains, and
  argument alone does not raise expected correctness. [H] Their "heterogeneous" agents were
  personas on one model, not mixed vendors, so the paper does not support claims that mixing
  vendors helps.
- A re-test of 5 debate frameworks on 9 benchmarks found debate generally failed to beat
  self-consistency (several samples plus a vote). Debate turned correct answers wrong more often
  ([ICLR 2025 blogpost](https://iclr-blogposts.github.io/2025/blog/mad/)). [H]
- In fairness: the original debate paper (Du et al.) did beat a majority-vote baseline, and tuned
  debate protocols can win ([Smit et al.](https://github.com/instadeepai/DebateLLM)). [H]
- In finance, a debate agent did not beat the best single agent
  (ΔSharpe −0.004, p = 0.77; [Macro Economists in the Machine](https://arxiv.org/abs/2606.08283)).
  The debate's value there was averaging out one agent's miscalibrated prior. [H]
- The quality of agents' conversation had zero correlation with returns
  ([2511.13614](https://arxiv.org/abs/2511.13614)). [H]

**Quality beats diversity, and diversity has to be measured.**

- Aggregating many samples of the single best model beat mixing models by 6.6% on AlpacaEval
  ([Self-MoA](https://arxiv.org/abs/2502.00674)). [H]
- A 2024 three-vendor LLM crowd on ForecastBench scored below its best member. [H]
- Across 350+ models, two models agree on about 60% of the questions they both get wrong, and
  stronger models' errors are more correlated, even across providers
  ([Kim et al., ICML 2025](https://arxiv.org/abs/2506.07962)). [H]
- LLM agents given identical evidence herd. Giving them disjoint private evidence and pooling
  confidence-weighted log-odds improved Brier scores by 12–18% (InfoDelphi, arXiv 2607.01661, a
  preprint). [M]

**Multi-agent systems fail for organisational reasons.** The MAST taxonomy (1,642 traces, v3)
attributes 44% of failures to system design and specification, 32% to inter-agent misalignment
and 24% to weak verification
([MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)). [H]

**When multiple agents pay off.**

- Anthropic's research system (an Opus 4 lead with Sonnet 4 subagents) beat a single agent by
  90.2% on an internal research task. Token spend explained 80% of the variance, and multi-agent
  systems used about 15× the tokens of chat
  ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). [H] This is
  breadth-first research, not tightly coupled decisions.
- Google's controlled study found centralised coordination helped a 50-question *financial
  research Q&A* benchmark by about 81%, which is not trading. Sequential planning got 39–70%
  worse. Independent agents amplified errors 17.2× vs 4.4× with a verifying orchestrator
  ([2512.08296](https://arxiv.org/abs/2512.08296)). [H]

**Personas do not add accuracy.** Across 162 roles and 2,410 questions, persona system prompts did
not improve accuracy
([Zheng et al.](https://github.com/Jiaxin-Pei/Prompting-with-Social-Roles)). [H]

**LLM judges are biased.** They show position, verbosity and self-preference bias. Swapping answer
order let a weaker model "win" 66 of 80 comparisons
([FairEval](https://github.com/i-Eval/FairEval)). [M]

**Reflection needs reliable feedback.** Verbal self-reflection
([Reflexion](https://github.com/noahshinn/reflexion)) works with unit tests. Without external
feedback, self-correction often fails. Market P&L is noisy and delayed. [M]

### 3.3 What to copy

| Copy this | From | Into our design as |
|---|---|---|
| Hard interfaces between research, PM, risk, execution, compliance | Every real firm; Rule 15c3-5 | Typed handoffs; risk and compliance in code that agents cannot change |
| Independent first, then statistical aggregation | Lorenz, Navajas, Good Judgment, Choi et al. | Sealed analyst forecasts pooled in log-odds |
| Debate only to surface private information | Navajas (small cells); hidden-profile research | A challenge round triggered by disagreement, then private revision |
| Authentic dissent | Nemeth 2001 | The challenger is the analyst whose own view most opposed the pool |
| Track-record weighting, shrunk toward equal | Bridgewater, Good Judgment, ai-hedge-fund v2 | Brier-based weights with a minimum sample, floor and cap; equal-weight shadow logged |
| Staged capital cuts and central risk | Pod platforms | Drawdown ladder; sleeve budgets; crowding monitor |
| Same gate for AI and human ideas | Man Group | No signal gets capital without out-of-sample and paper evidence |
| Decision journals, pre-mortems, kill criteria | Mauboussin, Klein, Pabrai | Machine-checkable IC memo frozen before execution |
| Sizing separate from picking; selling as carefully as buying | Best Ideas; Akepanidtaworn | Deterministic sizer; exits by the same engine and re-underwriting |

---

## 4. Failure modes and defences

| Failure | Evidence | Mitigation |
|---|---|---|
| **Look-ahead from the model's own memory** | GPT-4o recalls exact pre-cutoff S&P 500 levels and macro figures. Instructions to ignore later data do not stop it, and masking fails ([Memorization Problem](https://arxiv.org/abs/2504.14765)) [H]. Sharpe decays 51–62% after the cutoff (Profit Mirage) [H] | Count only post-cutoff results as evidence. Keep a registry of model cutoffs (Section 7.2). Run blind mode with a measured re-identification gate. Use the lookahead-propensity test for open models. |
| **Look-ahead in the data pipeline** | TradingAgents fixed leaks in FRED vintages, social posts, memory, fundamentals and date strings during 2026 [H]. ai-hedge-fund v1 filtered fundamentals by report period [H] | Use one point-in-time data gateway. Take dates from orchestrator state, never from the LLM. Test with planted leaks (rta-bench caught 48 of 48 when cuts fell on decision days) [H]. |
| **Survivorship and hindsight ticker choice** | Only 1 of 19 early audited studies handled it [H]. Papers favour today's mega-caps, which models remember best | Use a point-in-time universe that includes delisted names, chosen with prior-window information only. |
| **Short windows and many trials** | Median test window 12 months; 9% of papers multiplicity-adjusted [H]. A 3-month Sharpe has a standard error of about 2. The best of 10 skill-less one-year backtests is expected to show a Sharpe near 1.6 [M] | Keep a trial ledger. Use the deflated Sharpe ratio and minimum track record. Spend the holdout once. Size test windows with power analysis (Section 7.3). |
| **Ignored costs** | 51% of papers deduct none [H]. The GPT-4 news strategy is unprofitable at 20 bps round trip [H] | Use next-open fills, spread, commission, square-root impact and LLM token cost. Report the cost at which the edge breaks even. |
| **Beta mistaken for alpha** | 78% of returns explained by market and style; no better than random same-sector picks [H]. LiveTradeBench's "all positive" trailed equal-weight passive [M] | Run factor regressions. Compare with composition-matched random books, an equal-weight universe and a quant-only shadow book. |
| **Hallucinated numbers and entities** | FinanceBench: 81% wrong or refused (mostly refusals) with naive retrieval [H]. TradingAgents' wrong-company and invented-indicator bugs [H] | Every number cites a fact ID from the data desk. A grounding checker runs. Entity resolution is deterministic. Anything unsourced is tagged and excluded from sizing. |
| **Sycophancy and anchoring** | RLHF models bend toward stated beliefs ([Sharma et al.](https://arxiv.org/abs/2310.13548)) [H]. A quoted "Consensus: Buy" overrode TradingAgents' PM [H] | Keep the client or PM thesis and third-party ratings out of analyst prompts. Run counterfactual-framing tests. |
| **Herding in debate** | Agents switch from correct to wrong answers under peer pressure (Wynn et al.). Debate is a martingale (Choi) [H] | Sealed first views. Debate only on disagreement. Log opinion flips and the reason for each. |
| **Overconfidence** | Stated LLM confidence clusters at 80–100% ([Xiong et al.](https://arxiv.org/abs/2306.13063)). Post-training worsened GPT-4's calibration [H] | Score probabilities with Brier. Recalibrate (Platt or isotonic). Size from calibrated edges, never from stated confidence. |
| **Wrong-way risk appetite by regime** | Too cautious in bull markets, too aggressive in bear markets (FINSABER) [H] | Market exposure is set by code (volatility target, drawdown rules), never by LLM market calls. |
| **Leverage, overtrading, flip-flopping** | Alpha Arena losers [L]. Production fleets at 5× leverage regardless of volatility [H]. A nonsense-instruction agent made ~700 trades for +0.2% while passive returned +2.2% to +6% [H] | No LLM-chosen leverage. Turnover budget, no-trade band, minimum holding period. |
| **Run-to-run randomness and silent model updates** | Up to 15% accuracy variation at "deterministic" settings ([Atil et al.](https://arxiv.org/abs/2408.04667)). GPT-4's prime-number accuracy fell from 84% to 51% between versions ([Chen et al.](https://arxiv.org/abs/2307.09009)) [H] | Sample k times and use dispersion as uncertainty. Pin dated model IDs. Run a daily canary. Treat a model change as a new agent. |
| **Prompt injection via news, filings and web** | Frontier agents fail 10.7–29.6% of live indirect-injection cases ([LivePI](https://arxiv.org/abs/2605.17986)). Adaptive attacks bypass 12 published defences at >90% ([Nasr et al.](https://arxiv.org/abs/2510.09023)) [H] | Quarantined, tool-less readers that emit typed JSON. No raw text reaches privileged agents. No agent holds order tools (Section 5.5). |
| **Misinformation that looks true** | Three current LLMs sell whole positions on a fake headline (77–100% of decisions) and on the replayed Emulex hoax (100%) ([FARSIGHT](https://arxiv.org/abs/2609.19705)) [H] | Verification tiers, upstream-independent corroboration, a confirmation hold for single-source events, a rule-based price-shock freeze. |
| **Tool/MCP hijack** | A fake price feed drove an agent to put 78% of cash into a fake crash ([TradeTrap](https://github.com/Yanlewen/TradeTrap)). Tool-poisoning attack success 72.8% ([MCPTox](https://arxiv.org/abs/2508.14925)) [H] | Allowlist MCP servers and pin their descriptions by hash. Use two price vendors over different paths. |
| **Memory poisoning** | Memory injection: 98% injected, 77% attack success ([MINJA](https://arxiv.org/abs/2503.03704)) [H] | A code gate on memory writes. Lessons only from broker-verified outcomes and verified facts. Quarantine before promotion. |
| **Ledger/state tampering or desync** | A procedural agent lost 61% building a short it could not see (TradeTrap) [H] | Positions only from the broker. Reconcile every cycle. Halt on mismatch. |
| **Correlated failure across agents and firms** | Mixed-model agent populations still crashed a simulated market in 92–100% of runs [H] | Loss limits that halt rather than sell. Fast-path freeze. Data-source diversity. Firm-wide exposure limits. |
| **Alpha decay and crowding** | LLM news Sharpe fell from 6.54 to 1.22 in 2.5 years [H]. Published anomalies lose about 58% post-publication ([McLean & Pontiff](https://doi.org/10.1111/jofi.12365)) [M] | Monitor rolling IC per signal. Haircut backtests by about 50%. Re-validate on a schedule. |
| **Cost blow-up** | Multi-agent systems use about 15× chat tokens [H] | Model tiers, caching, a hard dollar budget per run, a cheap screen before deep dives. |

---

## 5. The data layer

### 5.1 Principles

1. **One gateway, point-in-time by construction.** Every agent tool goes through a single data
   service. It takes `as_of` from the orchestrator, clamps any date the model asks for, and
   returns typed, timestamped facts with IDs. The contract TradingAgents converged on is the
   template [H]:
   - date clamping inside each tool;
   - fundamentals served *as filed*;
   - macro data pinned to the vintage known on the date;
   - live-only data withheld from historical runs rather than leaked;
   - "unavailable" reported distinctly from "none found".
2. **Store facts bitemporally.** Each record carries:
   - value, unit and period;
   - **knowledge time**: SEC acceptance time, article publish or first-seen time, macro vintage,
     or bar close;
   - source, retrieval time and adjustment basis;
   - licence tag and quality flags.

   Point-in-time queries filter on knowledge time ≤ as-of.
3. **Fail loud.** An infrastructure error raises. "No data" means the data genuinely does not
   exist. Silent empty results create fake signals and fake silence. This follows ai-hedge-fund
   v2's contract. [H]
4. **No number without a fact ID.** Agents get compact fact tables. A checker matches every
   number in their prose to a fact or a deterministic computation. Anthropic's reference
   financial agents use the same rule, marking anything unsourced as `[UNSOURCED]`
   ([repo](https://github.com/anthropics/financial-services); only 3 of its 10 agents carry the
   rule). [H]

### 5.2 Source-by-source pitfalls

- **SEC EDGAR** ([APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)).
  Free, no key, but send a User-Agent with a name and email and stay at or below 10 requests per
  second. [H]
  - `companyfacts` tags every XBRL fact with its filing date and form. Filtering to filed ≤ as-of
    and taking the latest filing per period gives true as-filed fundamentals, restatements
    included. For example, Apple's FY2008 total assets read $39.6B until a 2010 amendment
    restated them to $36.2B. [H]
  - The `frames` endpoint returns the last-filed value, so it is **not** point-in-time. [H]
  - The `filed` field is a date only. A filing made after the close can leak into a same-day
    decision unless you use acceptance timestamps.
  - XBRL starts around 2009–2011.
  - Form 4 insider trades must be dated by filing date (up to 2 business days after the trade).
    13F filings are up to 45 days stale.
  - [edgartools](https://github.com/dgunning/edgartools) is a good parsing library.
- **Commercial fundamentals are usually not point-in-time.** Massive (formerly Polygon.io) says
  its values are the latest restated numbers. Its `filing_date` is the most recent filing that
  included the period. Its "adjusted" prices are split-only
  ([Massive skills](https://github.com/massive-com/ai)). [H]
- **Prices.**
  - yfinance (v1.8.0, Nov 2026) is unofficial and its README says "personal use only". Default
    prices are back-adjusted for splits and dividends, so history is rewritten after each
    dividend ([repo](https://github.com/ranaroussi/yfinance)). [H]
  - Store raw as-traded prices plus a separate corporate-actions table.
  - Report total return, not price return.
- **Macro.** FRED serves the *latest revision* by default. Pin `realtime_start = realtime_end =
  as_of` to get what was known then
  ([FRED docs](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)). [H]
- **News and social.**
  - Store both the published time and the first-seen time, and use the later of the two.
  - Join news to prices by publication timestamp, never by calendar date. Joining by calendar
    date made 5 of 22 event types look predictive. With publication-time replay, none were. [H]
  - GDELT's DOC API officially covers only the last 3 months.
  - Yahoo serves only recent news. Alpha Vantage's news endpoint serves dated historical
    windows. [H]
- **Survivorship-free universes.**
  - Massive's reference endpoint lists every stock on a given date, including names delisted
    later.
  - Beware ticker reuse: match on FIGI plus name. [H]
  - Free S&P 500 membership history ([fja05680/sp500](https://github.com/fja05680/sp500)) uses
    Clenow's dataset for 1996–2019 plus manual updates since. Verify it before relying on it.
  - Norgate, Sharadar or CRSP are the paid upgrades.
- **Retrieval alone is not enough.** In FinanceBench, GPT-4-Turbo with a shared vector store got
  81% of filing questions wrong or refused them (68 points were refusals). Better retrieval
  reached 50–79% correct ([paper](https://arxiv.org/abs/2311.11944)). [H]

### 5.3 Recommended starter stack

| Tier | Prices | Fundamentals / events | Macro | News | Universe | Notes |
|---|---|---|---|---|---|---|
| **0 – research, personal account ($0)** | Alpaca (IEX real-time, 15-min-delayed SIP history), yfinance only as a cross-check | SEC EDGAR via edgartools (as filed, Form 4, 8-K) | FRED/ALFRED (vintage-pinned), BLS, Treasury | Alpha Vantage free (25 requests/day), GDELT | fja05680 list + security master | Parquet/DuckDB store, immutable once a period closes |
| **1 – serious paper trading (~$30–150/mo)** | One licensed end-of-day vendor with delisted names (e.g. Tiingo or EODHD), Alpaca SIP (~$99/mo per its archived docs) | EDGAR, plus a paid vendor as validator | Same | Paid news with timestamps | Vendor delisted lists | Vendor prices not re-verified for Nov 2026 [L] |
| **2 – before real capital at scale** | Norgate or Sharadar | Point-in-time fundamentals vendor | Same | Archived timestamped newswire | Point-in-time index membership | Options history: ThetaData/ORATS if needed |
| **3 – managing outside money** | Enterprise feeds | Institutional connectors (FactSet, S&P/Kensho, LSEG, Daloopa, Aiera via MCP) | Same | Same | Same | Enterprise pricing |

**Licensing matters earlier than people expect.** By Massive's definition, "non-professional"
means a natural person using data for personal, non-business use. A firm entity using data for
business is professional from day one, whether or not it manages outside money. Yahoo data is
"personal use only". [H] The current project uses Yahoo prices. That is acceptable for personal
research but must be replaced before any business use.

**Validation on ingest.** FinRobot's thresholds are a reasonable start [H]:

- prices: 5% hard reject across vendors (warn at about 1%);
- revenue and net income: 15%;
- cash: 10%;
- market cap: 5%;
- prose numbers vs tables: 1%.

Also flag stale quotes, and check that a split-adjusted series rebuilt from raw bars matches the
vendor's.

### 5.4 MCP servers

MCP is now the standard interface. Examples include OpenBB, the Alpha Vantage remote MCP,
Financial Datasets, Massive and Alpaca. Most of these servers pass vendor data through unchanged,
with no as-of control. The Financial Datasets fundamentals tools take no date filter at all [H].
Use vendor MCPs for interactive research. Put one in-house data-gateway MCP in front of them for
anything that feeds decisions or backtests.

### 5.5 Treat external text as hostile

Data-gathering agents are the most exposed part of the firm. Anyone can plant text in news
bodies, social posts, filings HTML, transcripts and third-party tool output. The defensible
pattern, drawn from CaMeL ([paper](https://arxiv.org/abs/2503.18813)), Anthropic's reference
readers and FARSIGHT:

1. **Quarantined readers.** Only tool-less, memory-less reader calls ever see raw text. Each
   handles one document and outputs strict JSON:
   - entity IDs validated by code;
   - an event type from a fixed list;
   - numbers with source-span offsets;
   - tiny capped free-text fields.

   Code, not the LLM, attaches metadata such as source, time and hash.
2. **Deterministic verification tiers.**
   - Tier A: confirmed on a primary channel (issuer site, an EDGAR 8-K from a filer with history,
     a regulator or exchange).
   - Tier B: two *upstream-independent* reputable outlets. Syndicated copies count once.
   - Tier C: everything else, including posts from official or verified-looking social accounts.
     The AP and SEC hoaxes came through hacked official accounts, and the Eli Lilly hoax through an
     impostor "verified" account. [M]
3. **Only Tier A/B facts can open or add to positions.** A Tier C event triggers an alert and a
   hold of up to 60 minutes. Documented hoaxes reverted within minutes to about an hour, and a
   TradingAgents-style decision loop already takes more than 4 minutes, so the hold costs little.
   [M]
4. **A rule-based fast path.** Freeze new orders when price falls more than 3 calm-period sigmas
   within 60 seconds. Halt, do not liquidate, an agent that breaches a loss limit. In FARSIGHT's
   simulation, a stop-loss that *sells* turned 400 of 450 shocks into collapses. [H]

---

## 6. Turning views into a portfolio

### 6.1 What agents should output

Each analyst should emit a structured, resolvable forecast, never a trade or weight:

- the probability that the stock beats the benchmark over the horizon;
- expected excess return;
- bull, base and bear scenarios with probabilities;
- horizon;
- evidence cited by fact ID;
- what the analyst knows that others likely do not;
- an explicit abstain option.

Abstaining is different from neutral. A failed call becomes an abstention, never a default
position. ai-hedge-fund v1's generic fallback picked the first allowed value, which was
"bullish". [H]

**Calibrate expectations.** For a one-month horizon, honest probabilities mostly sit between 0.35
and 0.65. On ForecastBench market questions, simply copying the market price is a very hard
baseline. The value is in small, well-calibrated deviations from the prior. [H]

### 6.2 Aggregation

1. **Remove each agent's directional bias.** LLMs show persistent optimism or pessimism. In one
   study one model's average view was negative throughout
   ([LLM-Enhanced Black-Litterman](https://arxiv.org/abs/2504.14345)). They also prefer large-cap
   tech ([Lee et al.](https://arxiv.org/abs/2507.20957)). [H] Standardise each agent's scores
   across stocks on each date (median and MAD), capped at ±3.
2. **Pool robustly.** Use a trimmed mean or median of log-odds across agents and across k = 3–10
   samples per agent. Halawi et al.'s forecasting system halves the weight of the most outlying
   forecast ([repo](https://github.com/dannyallover/llm_forecasting)). [H]
3. **Weight by track record only with shrinkage.**
   - Start at equal weights.
   - After at least about 20 graded dates per agent, tilt weights by the t-statistic of each
     agent's IC or Brier skill, with a floor and a cap. ai-hedge-fund v2 uses
     `clip(1 + 0.25·t, 0.25, 2.0)`. [H]
   - Always log the equal-weight result alongside.
   - Reset the record when an agent's underlying model changes.

   The evidence is mixed. ContestTrade's contests mattered (with the caveat in 2.2), while
   FinPanel's track records changed nothing.
4. **Extremize** (push the pooled probability away from 0.5) only if fitted out-of-sample. It
   assumes agents hold independent information, which same-model agents often do not.
5. **Blend with market-implied priors.** Bridgewater's forecaster plus market consensus beat
   consensus alone. [H]

### 6.3 From probabilities to expected returns to weights

- **Grinold scaling** keeps alphas realistic: expected residual return = IC × residual volatility
  × z-score. With IC 0.05 and 25% annual residual volatility, a one-standard-deviation name
  expects only about +0.25% over two weeks.
- **Black-Litterman** is the natural Bayesian bridge from views to weights. One published recipe
  takes 100 samples per stock, uses the sample mean as the view and the sample variance as the
  view uncertainty, then runs mean-variance optimisation. Its best model reached Sharpe 1.23 vs
  0.89 for equal weight over 10 months on 50 mega-caps. That gap is not statistically
  significant, and the universe was chosen mid-test. [H]

  Floor the view uncertainty, because sampling spread understates true error. Inflate it by each
  agent's realised forecast error.
- **Recommended v1:** take the top K names (20–30 in a broad universe) by pooled score, weight by
  inverse 60-day volatility, apply caps, and size the total by a volatility target. Equal weight
  is hard to beat when expected returns are noisy
  ([DeMiguel et al. 2009](https://doi.org/10.1093/rfs/hhm075)). [M] Move to Black-Litterman plus
  an optimiser only after IC is proven.
- **Kelly:** use at most a quarter to a half. Half Kelly gives about 75% of the growth at half
  the volatility. Full Kelly with estimated edges is ruinous. [M]
- **Long-only asymmetry.** LLM ranking information concentrates in the top tier. Bottom-ranked
  stocks were indistinguishable from the market
  ([Chen & Pu](https://arxiv.org/abs/2601.11958)) [M]. Let bearish views reduce or avoid a name.
  Don't open shorts until a short-side IC is proven.

### 6.4 Risk limits (suggested defaults for a small long-only book)

These are engineering defaults drawn from open-source practice (ai-hedge-fund v2, Track-or-Follow,
Vibe-Trading) and the drawdown literature. They are not optimised values.

| Limit | Default |
|---|---|
| Single name | ≤ 8% target, 10% hard cap |
| Sector | ≤ 25% absolute, or within ±10% of benchmark |
| Ex-ante beta | 0.85–1.10 (Blume-adjusted) |
| Volatility target | 12–15% annualised; excess goes to T-bills; no leverage |
| Order size | ≤ 5% of 20-day average daily volume |
| Fund drawdown | −10% from high-water mark halves gross; −20% → cash or benchmark, frozen until human review; re-risk after recovering half the drawdown |
| Sleeve/agent drawdown | −5% halves allocation; −10% freezes |
| Loss limits | Halt the agent; never auto-liquidate into a shock |
| Price stops | No tight stops on single stocks. A −15% or 3-sigma move forces a re-review instead. Stops add value only when returns trend ([Kaminski & Lo](https://dspace.mit.edu/handle/1721.1/114876)) |
| Turnover | ≤ 25% one-way per rebalance; 0.5%-of-equity no-trade band; move 50% of the way to target |
| Instruments | Cash equities and ETFs only until counsel signs off on derivatives |

Drawdown rules contain risk. They do not create alpha. A fixed threshold will sometimes cut a good
but unlucky strategy, so scale thresholds to each sleeve's expected volatility
([Man Institute](https://www.man.com/maninstitute/youre-fired)). [M]

### 6.5 Costs and turnover

Model each fill as:

- half-spread (about 5 bps for large caps);
- plus commission (about $0.005/share);
- plus square-root impact, about 1.0 × daily volatility × √(order / average daily volume).

Charge LLM token costs too. FINSABER-2 adds them to total trading cost.

Daily news-driven strategies are fragile:

- GPT-4's news strategy needed costs at or below 10 bps round trip.
- Rebalancing only 25% of the way to target each day cut turnover from 190% to 46% a day and gave
  a slightly better Sharpe at 10 bps. [H]
- By one estimate, daily equal-weighted news long-short books break even at roughly 6–8 bps
  one-way. [M]

A small fund should use monthly or weekly cross-sectional selection, not daily news trading.

### 6.6 Hybrid design: quant core plus bounded LLM overlay

The best-supported structure is a factor or passive core with the LLM as a bounded residual
overlay. The core is momentum, value, profitability, investment and low volatility, ideally with
a gradient-boosted model.

- **Why a core.** Published predictors keep only about 50–60% of their in-sample returns after
  publication ([Chen et al.](https://arxiv.org/abs/2212.10317)) [H], and net of costs the average
  anomaly is thin. But a reproducible baseline exists: qlib's LightGBM on Alpha158 features
  reached an IR of 1.02 net of costs on CSI 300 for 2017–2020
  ([qlib](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)). [H]
- **Why bounded.** Track-or-Follow fitted text features only on the residual of trusted features,
  clipped to twice the cross-sectional spread. Text was allowed only to reorder names inside the
  top band, never to promote low-ranked ones. At a weekly horizon text added **no** rank
  information: ΔIC −0.016, with a 95% interval that excludes zero
  ([2611.10414](https://arxiv.org/abs/2611.10414)). [H] Bounding made that harmless.
- **Blending helps.** In the Expert Investment Teams study, agent signals correlated about 0.4
  with the index, and blending raised Sharpe after costs (secondary figures). [M]
- **Keep a quant-only shadow book** on the same constraints, and let measured net-of-cost IC
  improvement set the overlay's weight.

---

## 7. Evaluation: how to test honestly

### 7.1 Why most published backtests are not trustworthy

The typical LLM trading paper:

- tests for 12 months or less;
- uses a few famous mega-caps;
- covers a period the model has already read about;
- charges no costs;
- reports a Sharpe without counting how many prompts, models and settings were tried. [H]

Each of these flaws inflates results on its own, and together they can manufacture a Sharpe of 3–8
from nothing. Statistical deflation does not catch leakage: the leaky oracle with Sharpe 35
passed. [H]

### 7.2 The contamination problem and what can and cannot be backtested

- **Text-reading agents cannot be backtested before their model's cutoff.**
  - GPT-4o named the firm behind anonymised earnings calls in 100% of AAPL, META and MSFT calls
    and 31% of small-cap calls.
  - It dated sets of WSJ headlines to the correct year 98.5% of the time. [H]
  - Before their cutoffs, frontier models dated a masked monthly macro snapshot to the right year
    95–100% of the time. GPT-5.5 still managed 94% after its cutoff
    ([HindsightBench](https://arxiv.org/abs/2607.18867)). [H]
- **Price/factor-only payloads can be blinded.** With masked price and factor inputs, ten
  frontier attackers named the right ticker on the first guess at most 3% of the time, and ticker
  plus date at most 1.5% ([KTD-Fin](https://arxiv.org/abs/2605.28359)). [H]
- **Vendor cutoffs are unreliable.** Measured effective cutoffs span 2023-12 to 2025-10 and
  differ from vendor dates by up to 8 months (GPT-5.5: measured 2025-06 vs reported 2025-12). [H]
  Start each model's clean window at the later of the two dates, plus one month.
- **Point-in-time models are small.** ChronoGPT has 1.55B parameters and a 1,792-token context
  ([paper](https://arxiv.org/abs/2502.21206)). They work as leak-free lower bounds for text
  signals, but cannot run an agentic firm. [H]
- **Lookahead propensity test.** For open-weight models, this test detects contamination: on
  headline forecasts, memorisation inflated in-sample effects by about 32%
  ([2512.23847](https://arxiv.org/abs/2512.23847)). [H]
- **Re-identification threshold.** No published standard exists. A derived gate is: on at least
  400 probes of the exact rendered payload, joint re-identification no more than 1 percentage
  point above chance, with a 95% upper bound of at most 2%. This keeps IC bias at or below about
  0.005. [M]

### 7.3 Statistical power: how long until you know?

**Minimum track record** (daily data, 95% one-sided, Bailey & López de Prado
[DSR/MinTRL](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)) [H]:

| True annual Sharpe | Years of live track record needed |
|---|---|
| 0.5 | ≈ 10.8 |
| 1.0 | ≈ 2.7 |
| 1.5 | ≈ 1.2 |
| 2.0 | ≈ 0.7 |

**Signal-level IC tests are much faster**, if the universe is broad. You need about (2.80 / IC)²
stock-dates for 80% power at the 5% level. With a realistic per-date IC volatility floor (0.06),
that becomes [M]:

| IC | Daily dates, 500 names | Daily dates, 50 names | Monthly, 10 names (independence approximation; real figure is longer) |
|---|---|---|---|
| 0.02 | ≈ 110 | ≈ 471 | ≈ 160 years |
| 0.03 | ≈ 49 | ≈ 209 | ≈ 70 years |
| 0.05 | ≈ 18 | ≈ 75 | ≈ 26 years |

Telling two architectures apart is harder still. Detecting an IC difference of 0.01 needs about
94–137 daily dates at 500 names and about 1,000 at 50 names. **A narrow monthly book of about 10
names cannot statistically validate skill or compare designs on any realistic horizon.** This is
the most important practical constraint on our design (Section 9).

Seeds reduce LLM sampling noise but add no market information: the date is the unit of
replication. Two seeds of the same model ended an average of 5.2 points of total return apart in
rta-bench. [H]

### 7.4 The protocol (staged gates)

- **S0. Machinery, no LLM.** Validate the data layer, costs, risk engine and statistics using stub
  agents.
  - Plant at least 4 leak types, 12 instances each; all must be caught.
  - Replay with the future randomised after cuts placed on decision days; decisions before each
    cut must not change.
  - Ledgers must replay exactly.
  - A planted IC of 0.02–0.03 must be recovered.
  - Zero limit breaches on stress paths.
  - Results must survive 2× costs.
- **S1. Contamination audit per model.** Establish the measured cutoff, the clean window and the
  pinned serving configuration.
- **S2. Re-identification gate** before any backtest on pre-cutoff data.
- **S3. Pre-cutoff development** may tune only plumbing, schemas, risk and costs. Never tune
  prompts or debate settings against pre-cutoff returns of text-reading agents.
- **S4. Architecture selection on a "vintage ladder".** Test each candidate design on at least 3
  models with staggered cutoffs, each on its own clean window, with at least 3 seeds. Then:
  - compare designs with paired per-date IC-difference tests;
  - default to the simpler design unless the difference reaches t ≥ 2 with a consistent sign;
  - require IC t ≥ 3 if a pick came from more than 20 trials;
  - require deflated Sharpe ≥ 0.95 and PBO ≤ 0.2 (a heuristic);
  - require factor alpha t ≥ 2 and a 95th percentile against composition-matched random books.

  There is **no published evidence** that rankings transfer from older models to newer ones.
  This ladder is the best proxy, not a proven method.
- **S5. One-use holdout** on the production model's window between its cutoff and its release.
- **S6. Forward paper trading** of one frozen configuration:
  - at least 126 trading days plus IC t ≥ 2.5 for broad daily books;
  - at least 2 years for books of 100 names or fewer, or for weekly or monthly horizons;
  - then pilot capital of at most 10% of target;
  - scale only after the minimum track record is met with every watched variant deflated.

**Every report states six lines** ([2611.18934](https://arxiv.org/abs/2611.18934)):

1. dated window and number of assets;
2. model and its cutoff, and whether the test lies after it;
3. cost model and breakeven cost;
4. number of configurations tried;
5. deflated statistic at that count;
6. a benchmark or composition-matched null.

Add factor attribution (Fama-French 5 plus momentum, or ETF proxies), results by regime, and an
equal-weight-pool vs weighted-pool comparison.

**Red-team in CI.** Use TradeTrap's injection, fake-news, fake-MCP and position-poisoning payloads
in a sandbox, plus renamed real hoaxes. Score attack success per decision, turnover multiple,
maximum concentration and orders reaching the broker (must be zero for blocked classes). Do not
use P&L: under one injected attack, one model's return *improved* to +20.5%. [H]

---

## 8. Practical and legal path (US-centric; not legal advice)

**Whose money it is decides almost everything.**

- Trading **your own money** with AI agents is not advising others. It needs no adviser
  registration ([Advisers Act §202](https://www.law.cornell.edu/uscode/text/15/80b-2)). [H] The
  constraints are broker rules, tax and data licences.
- Advising **others for pay**, pooling money or selling signals brings in federal or state
  adviser law. The old "fewer than 15 clients" exemption was repealed in 2011. [H]
- Below about $100M of assets an adviser generally registers with its state. SEC registration is
  allowed at $100M and required at $110M. Advisers solely to private funds under $150M can be
  "exempt reporting advisers" but still file parts of Form ADV. [H]

**A fund** typically means:

- a Delaware LP, a GP entity and a management company;
- an Investment Company Act exemption (3(c)(1): 100 or fewer owners; or 3(c)(7): qualified
  purchasers);
- a Reg D offering: 506(b) without advertising, or 506(c) with advertising but verified
  accredited investors;
- Form D within 15 days of the first sale. [H]

Performance fees require "qualified clients". The thresholds rose to **$1.4M with the adviser or
$2.7M net worth from June 29, 2026** (SEC Order
[IA-6961](https://www.sec.gov/files/rules/ia/2026/ia-6961.pdf)). A September 2026 proposal would
let accredited investors qualify; it is not adopted. [H]

**Marketing is the most likely early trap.** Under SEC Marketing Rule 206(4)-1, backtests and
paper results shown in an adviser's advertisements are "hypothetical performance". They require
audience-relevance policies and full disclosure of assumptions. In September 2023 the SEC
penalised 9 advisers a combined $850k over hypothetical performance on websites. In March 2024 it
brought its first "AI-washing" cases: Delphia paid $225k and Global Predictions $175k. [M] The
SEC withdrew its AI-specific "predictive data analytics" proposal on June 12, 2025. Existing
fiduciary, compliance, record-keeping and anti-fraud duties apply in full. [H] Never publish
backtest returns or "AI alpha" claims before counsel reviews them.

**Brokerage rules are changing.**

- The SEC approved FINRA's replacement of the $25,000 pattern-day-trader minimum with intraday
  margin requirements on April 14, 2026 (SR-FINRA-2025-017). It takes effect 45 days after
  FINRA's notice, with up to an 18-month phase-in, so some brokers may still apply the old rule.
  [H]
- Settlement is T+1.
- Paper trading at Alpaca is free and uses the same API. Its own docs say paper fills ignore
  market impact, slippage, queue position, fees and dividends. [H]
- Schwab's API has no paper trading, and its tokens expire after about 7 days, which forces
  weekly re-login. [H]
- Interactive Brokers has the broadest coverage but needs a running gateway process. [M]

**Tax.**

- Wash-sale losses are disallowed across *all* your accounts, including IRAs, where the loss is
  lost permanently. Brokers track this only within one account. [M]
- Short-term gains are taxed at ordinary rates.
- A high-turnover agent can lose most of a pre-tax edge, so report after-tax results against
  buy-and-hold. Consult a CPA.

**Operations borrowed from regulation.**

- Pre-trade checks in code, following Rule 15c3-5: notional caps, price collars, duplicate
  checks, allowlists.
- A kill switch that cancels all orders, as MiFID II RTS 6 requires.
- Model-risk governance per [SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm):
  an inventory, independent validation, and treating every prompt or model change as a new model.
- Generative-AI risk controls per [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf).
- Keep broker keys out of model context. Use separate paper and live keys and a dedicated
  limited-funds account.

**LLM running costs** (Anthropic list prices, Nov 2026
[pricing](https://platform.claude.com/docs/en/about-claude/pricing)) [H]:

- Opus 5.5: $4 / $20 per million input / output tokens.
- Sonnet 5.5: $2 / $10.
- Haiku 5.5: $0.10 / $0.50 for prompts up to 100K tokens.
- Cache reads cost about 0.05× the input price on Opus 5.5 and Sonnet 5.5.
- The Batch API is 50% off.

A TradingAgents-style graph uses about 22 calls, 220k input tokens and 50k output tokens per ticker
per day (an estimate). That costs about $0.05 on Haiku, $0.93 on Sonnet and $1.87 on Opus per
ticker-day. 50 tickers daily on Sonnet is about $1,000 a month before caching. [M] The `aifirm`
README measures about $2–6 per monthly cycle for 10 tickers on Opus 5.5. Put a cheap quantitative
screen in front of LLM deep dives for broad universes.

**Staged plan.**

| Stage | What | Gate to move on |
|---|---|---|
| 0. Research | Point-in-time data; S0–S3 tests; everything labelled hypothetical | All machinery gates pass |
| 1. Paper | ≥ 3 months for operations; longer for skill (Section 7.3); reconcile fills; kill-switch drills; track cost per decision | Operations clean; forward IC tracked |
| 2. Small real capital, own dedicated account | Hard notional caps; human approval of every order at first; tax/wash-sale module; after-tax benchmark | Pre-registered forward track meets the S6 gates |
| 3. Outside capital | Only with securities counsel and a CPA: choose state RIA, exempt-reporting adviser plus fund, or no outside money; Form ADV, compliance program, books and records, marketing review | Legal structure in place |

---

## 9. Recommended architecture and improvements over prior art

### 9.1 The design in one picture

```
Data gateway (code, point-in-time, bitemporal, licence-tagged, validated across sources)
   │  prices (raw + corporate actions) · SEC as filed · FRED vintages · 8-K events
   │  untrusted text → quarantined readers → typed JSON → verification tiers A/B/C
   ▼
Fact sheet per ticker (every number has an ID and a knowledge time)
   ▼
Analysts, each on a different data slice (sealed, k samples, blind mode in backtests)
   fundamental · technical · macro/regime · events (Tier A/B facts only) · quant factor model (code)
   → probability of beating benchmark, E[excess], scenarios, evidence = fact IDs, private insight, abstain
   ▼  grounding check (unsupported numbers flagged; flagged views down-weighted)
Pool (code): per-agent cross-sectional de-biasing → trimmed log-odds mean
   weights = equal until ≥ min sample, then shrunk track-record tilt; equal-weight shadow always logged
   ▼  dispersion high? → challenge by the most-opposed analyst → private revisions → re-pool
IC memo (LLM, machine-checkable; can only shrink the edge): base rate, pre-mortem, kill criteria
   ▼
Portfolio engine (code): calibrated edge → Grinold/fractional Kelly → caps → vol target
   → core/overlay blend (LLM overlay bounded around the quant core) → no-trade band, turnover cap
   ▼  LLM risk review (can only scale down)
Invariant risk core (code; agents cannot change it): drawdown ladder, loss halt (no forced selling),
   price-shock fast path, liquidity caps, restricted list, kill switch, broker reconciliation
   ▼  human approval (live) → broker (paper by default) → hash-chained ledger
Scoring: forecasts resolve at horizon → Brier/IC per agent, model and regime → weights; structured lessons
   gated by point-in-time resolution · shadow books: quant-only, equal-weight, benchmark, random-matched
```

This matches the `aifirm` design in [docs/ARCHITECTURE.md](../ARCHITECTURE.md) (principles P1–P10)
and adds the refinements in 9.3.

### 9.2 Roles, data flow and protocol

**Roles.**

- *Data desk*: code only.
- *Readers*: tool-less LLM calls, one per document.
- *Analysts*: LLM, each defined by a task, a checklist and a data slice, not a persona.
- *Quant analyst*: code.
- *Challenger*: whichever analyst most opposed the pool.
- *Chair*: writes the IC memo and can only shrink the edge.
- *Risk reviewer*: LLM that can only scale positions down.
- *Portfolio engine and risk core*: code.
- *Scorer*: code.
- *Human*: approves live orders, unfreezes after drawdowns, approves changes to limits.

**Decision protocol.**

1. Sealed independent views.
2. Statistical pool.
3. Argument only on disagreement.
4. Private re-estimation.
5. Pooled again.

No LLM writes a consensus number. The chair and risk reviewer can reduce exposure but cannot add
it. Different rules can apply to different decisions. For example, one high-track-record analyst
could "champion" a small capped starter position in an under-covered name, while full size needs
the pooled signal. Risk and compliance vetoes are absolute.

**Cadence and universe.**

- Default to monthly or weekly cross-sectional selection.
- Do not time single stocks or trade intraday.
- For statistical power, cover a broad liquid universe (300–500 names) with a cheap screen. LLM
  deep dives go only to candidates and holdings.
- Alternatively, accept that a narrow book needs multi-year forward paper trading before any
  conclusion.

**Memory.**

- Lessons are structured records: claim, evidence, horizon, the date the outcome became known,
  and the model version.
- Retrieve only lessons resolved before the as-of date.
- Promote a lesson to policy only after repeated out-of-sample confirmation.
- Agents that read untrusted text never write memory.

### 9.3 Changes we recommend to the current design

1. **Breadth for power.** With about 10 names monthly, skill cannot be confirmed for decades
   (7.3). Add a broad screened universe, or state plainly in reports that results are anecdotal.
2. **Measured cutoffs and a re-identification gate.** Record measured as well as vendor cutoffs
   (HindsightBench-style). Before any pre-cutoff backtest, run a ≥ 400-probe re-identification
   test on the exact output of `blind_factsheet`.
3. **Quarantine headlines.** The events analyst should see reader-extracted, tier-labelled event
   records, not raw headlines. Only Tier A/B events may raise exposure.
4. **Licensed data before business use.** Replace Yahoo prices with a licensed vendor that
   includes delisted names. Store raw prices plus corporate actions.
5. **Equal weights by default.** Track-record weights only after a minimum sample, with shrinkage.
   Measure pairwise error correlation between analysts and down-weight redundant ones.
6. **Stronger baselines in the backtest report.** Add, where missing, a factor regression, 1,000
   composition-matched random books, and the quant-only shadow book's IC next to the LLM pool's.
7. **Red-team suite in CI.** Use injection, fake-news, fake-data and ledger-tamper fixtures,
   scored on orders reaching the broker.
8. **Correct the evidence cited for P7.** ContestTrade's 3.12 → 0.07 ablation is not evidence for
   track-record weighting alone (2.2). FinPanel is a null result.

### 9.4 Specific improvements over prior art

| # | Improvement | Over what | Evidence |
|---|---|---|---|
| 1 | LLMs never set sizes, leverage or limits; a deterministic shrink-only risk core with a halt-not-sell loss limit and a price-shock fast path | TradingAgents' LLM-only risk debate; AutoHedge; Alpha Arena agents choosing leverage | FinCon CVaR ablation [H]; FARSIGHT fast path 9.6% → 1.1% and sell-stops causing collapse [H]; Barton: volatility-blind 5× leverage [H] |
| 2 | Point-in-time gateway with planted-leak CI tests and fail-loud semantics | TradingAgents before 2026; ai-hedge-fund v1 | TradingAgents CHANGELOG leaks [H]; rta-bench 48/48 planted leaks caught [H]; leaky Sharpe-35 oracle passes DSR/PBO [H] |
| 3 | Sealed independent forecasts, statistical pooling, debate only on disagreement, authentic challenger | Fixed bull/bear debates; consensus-by-conversation | Choi et al. [H]; Macro Economists ΔSharpe −0.004 [H]; Lorenz, Navajas [H]; Nemeth [M] |
| 4 | Analysts defined by data slice and task, not persona | ai-hedge-fund personas; FinMem persona | Personas give no accuracy gain [H]; fine-grained tasks beat coarse roles [H]; private evidence +12–18% Brier [M] |
| 5 | Resolvable, calibrated probability forecasts scored by Brier, recalibrated, anchored on market priors | Buy/sell verdicts; stated confidence used as size | ForecastBench (market price 70.2; ensembles about superforecasters) [H]; AIA Forecaster [H]; LLM overconfidence [H] |
| 6 | Per-agent de-biasing, then shrunk track-record weights with an equal-weight shadow | ContestTrade's 5-day LightGBM; unweighted debate | LLM optimism/pessimism styles [H]; ContestTrade (corrected) [H]; FinPanel null [H]; ai-hedge-fund v2 scorecard [H] |
| 7 | Quant core plus bounded LLM residual overlay, with a quant-only shadow book | Pure LLM books | Track-or-Follow (text added nothing, bound kept it harmless) [H]; qlib baseline IR 1.02 [H]; post-publication decay [H] |
| 8 | Contamination controls: measured cutoffs, re-identification gate, post-cutoff-only evidence, vintage ladder | 98% of at-risk papers with no control | HindsightBench [H]; KTD-Fin [H]; Memorization Problem [H]; Profit Mirage [H] |
| 9 | Evaluation with trial ledger, deflated Sharpe, minimum track record, power analysis, factor attribution, random-matched nulls, six-line reports | 12-month median windows, no costs, no trial counts | Short-windows audit [H]; Gencay [H]; Policy-Value Audit ([2610.04040](https://arxiv.org/abs/2610.04040)) [H] |
| 10 | Quarantined readers, verification tiers, independent-source quorum, confirmation hold | Agents reading raw web text with trade tools (AI-Trader) | TradeTrap [H]; FARSIGHT [H]; Poisoning Agentic Alpha (97% when the final LLM node is jailbroken) ([2608.24069](https://arxiv.org/abs/2608.24069)) [H]; CaMeL [H] |
| 11 | Memory written only through a point-in-time code gate | FinMem layered memory; TradingAgents before fix #1251 | MINJA, AgentPoison [H]; TradingAgents memory leak fix [H] |
| 12 | Cost engineering: content-hash cache, model tiers, dollar budget halts, LLM cost counted as trading cost | Daily full-graph calls | ai-hedge-fund v2 cache [H]; FINSABER-2 LLM cost accounting [H]; 15× token multiplier [H] |
| 13 | Hash-chained ledger, reconcile-before-trade, kill switch, human approval, separate paper and live keys | Ad-hoc logs | ai-hedge-fund v2 (paper-only so far) [H]; Vibe-Trading order guard [H]; Rule 15c3-5 / Knight [M] |

---

## 10. Open questions

1. Does a full multi-agent pipeline beat a matched single agent with identical tools, data and
   costs, after the cutoff and net of token and trading costs? No controlled study answers this.
2. Can LLM agents produce significant stock-selection alpha after controlling for market, style
   and sector over multi-year post-cutoff windows? Current evidence is negative or inconclusive.
3. Which information channel carries signal? Technical drove one Japanese study, fundamentals led
   MarketSenseAI on the S&P 500, and removing news changed nothing in rta-bench. These differ by
   horizon, universe and prompt design, so measure rather than assume.
4. Do architecture rankings found on older models carry over to newer frontier models? No
   evidence either way. The vintage ladder (7.4) is a proposal.
5. How many resolved forecasts are needed before track-record weighting beats equal weighting,
   and should records carry over across model upgrades?
6. What extremizing, if any, is right for agents that share training data?
7. Does reflection memory improve out-of-sample results or mainly overfit recent luck?
8. At what rebalance frequency and liquidity do LLM coordination costs break even ("Coordination
   Breakeven Spread", [2603.27539](https://arxiv.org/abs/2603.27539))?
9. How do you measure upstream independence of news sources, and resist an attacker who creates
   many cheap "independent" sources?
10. Is the FINRA PDT replacement in effect at your broker yet? Which state's adviser rules apply
    to you? Do your data vendors' licences allow sending their data to third-party LLM APIs?
11. Unverified items to re-check before relying on them:
    - Alpha Arena S1/S1.5/S2 per-model results;
    - vendor prices for Tiingo, EODHD, Finnhub and Alpaca SIP;
    - the Scherer committee results;
    - the Millennium pod thresholds;
    - Essentia and Alpha Theory sizing figures;
    - the Atanasov, Mandel and Nemeth abstracts (consistent with recollection but not re-fetched);
    - Expert Investment Teams' secondary Sharpe figures.
12. Several key November 2026 preprints (rta-bench-1, the 91-paper audit, Hard to Fool, FinPanel,
    Track-or-Follow) are recent and some are single-author. Watch for replications.

---

## 11. References

**Frameworks and repos.**
TradingAgents [paper](https://arxiv.org/abs/2412.20138), [repo](https://github.com/TauricResearch/TradingAgents), [CHANGELOG](https://github.com/TauricResearch/TradingAgents/blob/main/CHANGELOG.md) ·
FinCon [NeurIPS](https://proceedings.neurips.cc/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html) ·
FinMem [arXiv](https://arxiv.org/abs/2311.13743) ·
FinAgent [arXiv](https://arxiv.org/abs/2402.18485) ·
HedgeAgents [arXiv](https://arxiv.org/abs/2502.13165) ·
QuantAgents [arXiv](https://arxiv.org/abs/2510.04643) ·
AlphaAgents [arXiv](https://arxiv.org/abs/2508.11152) ·
ContestTrade [arXiv](https://arxiv.org/abs/2508.00554) ·
MarketSenseAI [1.0](https://arxiv.org/abs/2401.03737), [2.0](https://arxiv.org/abs/2502.00415), [2026](https://arxiv.org/abs/2604.17327) ·
Expert Investment Teams [arXiv](https://arxiv.org/abs/2602.23330) ·
ai-hedge-fund [repo](https://github.com/virattt/ai-hedge-fund) ·
Vibe-Trading [repo](https://github.com/HKUDS/Vibe-Trading) ·
FinRobot [repo](https://github.com/AI4Finance-Foundation/FinRobot), [whitepaper](https://arxiv.org/abs/2405.14767) ·
R&D-Agent(Q) [arXiv](https://arxiv.org/abs/2505.15155), [repo](https://github.com/microsoft/RD-Agent) ·
AlphaAgent [arXiv](https://arxiv.org/abs/2502.16789) ·
Alpha-GPT [arXiv](https://arxiv.org/abs/2308.00016) ·
DeepFund [arXiv](https://arxiv.org/abs/2505.11065) ·
Trading-R1 [arXiv](https://arxiv.org/abs/2509.11420) ·
FLAG-Trader [arXiv](https://arxiv.org/abs/2502.11433) ·
TradingGroup [arXiv](https://arxiv.org/abs/2508.17565) ·
QuantAgent [arXiv](https://arxiv.org/abs/2509.09995) ·
TradExpert [arXiv](https://arxiv.org/abs/2411.00782) ·
StockAgent [arXiv](https://arxiv.org/abs/2407.18957) ·
FinPanel [arXiv](https://arxiv.org/abs/2611.18150) ·
Track or Follow [arXiv](https://arxiv.org/abs/2611.10414) ·
Inside the Trading Loop [arXiv](https://arxiv.org/abs/2611.11657) (its paper-account result used deterministic fallbacks, so it is not evidence for the LLM design) ·
TradingAgents-CN [repo](https://github.com/hsliuping/TradingAgents-CN) ·
AutoHedge [repo](https://github.com/The-Swarm-Corporation/AutoHedge) ·
Agentic Trading Lab [repo](https://github.com/Open-Finance-Lab/AgenticTrading) ·
Dexter [repo](https://github.com/virattt/dexter) ·
Anthropic financial-services [repo](https://github.com/anthropics/financial-services)

**Benchmarks and evaluations.**
FINSABER [arXiv](https://arxiv.org/abs/2505.07078), [repo](https://github.com/waylonli/FINSABER) ·
Profit Mirage [arXiv](https://arxiv.org/abs/2510.07920) ·
Alpha Illusion [arXiv](https://arxiv.org/abs/2605.16895) ·
rta-bench-1 / Earn Beta Not Alpha [arXiv](https://arxiv.org/abs/2611.19530) ·
Short Windows audit [arXiv](https://arxiv.org/abs/2611.18934) ·
Xia et al. audit [arXiv](https://arxiv.org/abs/2605.19337) ·
Execution realism review [arXiv](https://arxiv.org/abs/2606.08285) ·
What survives honest evaluation [arXiv](https://arxiv.org/abs/2608.27734) ·
Hard to Fool [arXiv](https://arxiv.org/abs/2611.06607) ·
Production fleets (Barton et al.) [arXiv](https://arxiv.org/abs/2609.05663) ·
Agent Policy-Value Audit [arXiv](https://arxiv.org/abs/2610.04040) ·
LiveTradeBench [arXiv](https://arxiv.org/abs/2511.03628) ·
AI-Trader [arXiv](https://arxiv.org/abs/2512.10971), [repo](https://github.com/HKUDS/AI-Trader) ·
StockBench [arXiv](https://arxiv.org/abs/2510.02209) ·
Agent Market Arena [arXiv](https://arxiv.org/abs/2510.11695) ·
InvestorBench [ACL](https://aclanthology.org/2025.acl-long.126/) ·
Nof1 Alpha Arena [site](https://nof1.ai) ·
ForecastBench [data](https://github.com/forecastingresearch/forecastbench-datasets) ·
Coordination Primacy [arXiv](https://arxiv.org/abs/2603.27539) ·
Market-Dependent Communication [arXiv](https://arxiv.org/abs/2511.13614) ·
Macro Economists in the Machine [arXiv](https://arxiv.org/abs/2606.08283)

**Look-ahead and memorisation.**
Memorization Problem [arXiv](https://arxiv.org/abs/2504.14765) ·
LAP test [arXiv](https://arxiv.org/abs/2512.23847) ·
Glasserman & Lin [arXiv](https://arxiv.org/abs/2309.17322) ·
ChronoBERT/ChronoGPT [arXiv](https://arxiv.org/abs/2502.21206) ·
ChronoGPT-Instruct [arXiv](https://arxiv.org/abs/2510.11677) ·
Look-Ahead-Bench [arXiv](https://arxiv.org/abs/2601.13770) ·
FinCAD [arXiv](https://arxiv.org/abs/2605.24564) ·
KTD-Fin [arXiv](https://arxiv.org/abs/2605.28359) ·
HindsightBench [arXiv](https://arxiv.org/abs/2607.18867) ·
Time Machine GPT [arXiv](https://arxiv.org/abs/2404.18543)

**LLMs as signals and quant baselines.**
Lopez-Lira & Tang [arXiv](https://arxiv.org/abs/2304.07619) ·
Chen & Pu agentic nowcasting [arXiv](https://arxiv.org/abs/2601.11958) ·
Chen, Kelly & Xiu [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687) ·
Kirtac & Germano [arXiv](https://arxiv.org/abs/2412.19245) ·
Kim, Muhn & Nikolaev (withdrawn Feb 2025) [arXiv](https://arxiv.org/abs/2407.17866) ·
Bybee [arXiv](https://arxiv.org/abs/2305.02823) ·
LLM-Enhanced Black-Litterman [arXiv](https://arxiv.org/abs/2504.14345) ·
LLM investment biases [arXiv](https://arxiv.org/abs/2507.20957) ·
Gu, Kelly & Xiu [RFS](https://doi.org/10.1093/rfs/hhaa009) ·
Post-sample predictability [arXiv](https://arxiv.org/abs/2212.10317) ·
McLean & Pontiff [JF](https://doi.org/10.1111/jofi.12365) ·
Novy-Marx & Velikov [RFS](https://doi.org/10.1093/rfs/hhv063) ·
Lazy Prices [JF](https://doi.org/10.1111/jofi.12885) ·
JKP factors [site](https://jkpfactors.com) ·
qlib benchmarks [repo](https://github.com/microsoft/qlib/tree/main/examples/benchmarks) ·
Kronos [arXiv](https://arxiv.org/abs/2508.02739) ·
LLMs for time series [arXiv](https://arxiv.org/abs/2406.16964) ·
FinBERT [repo](https://github.com/yya518/FinBERT)

**Multi-agent and forecasting research.**
Debate or Vote [arXiv](https://arxiv.org/abs/2508.17536) ·
Du et al. debate [site](https://composable-models.github.io/llm_debate/) ·
Liang et al. MAD [repo](https://github.com/Skytliang/Multi-Agents-Debate) ·
Smit et al. [repo](https://github.com/instadeepai/DebateLLM) ·
MAD re-evaluation [ICLR blog](https://iclr-blogposts.github.io/2025/blog/mad/) ·
Self-MoA [arXiv](https://arxiv.org/abs/2502.00674) ·
Correlated errors [arXiv](https://arxiv.org/abs/2506.07962) ·
ReConcile [arXiv](https://arxiv.org/abs/2309.13007) ·
MAST [repo](https://github.com/multi-agent-systems-failure-taxonomy/MAST) ·
Anthropic multi-agent research [post](https://www.anthropic.com/engineering/multi-agent-research-system) ·
Building effective agents [post](https://www.anthropic.com/engineering/building-effective-agents) ·
Scaling agent systems [arXiv](https://arxiv.org/abs/2512.08296) ·
Personas [repo](https://github.com/Jiaxin-Pei/Prompting-with-Social-Roles) ·
FairEval [repo](https://github.com/i-Eval/FairEval) ·
Reflexion [repo](https://github.com/noahshinn/reflexion) ·
Silicon crowd [arXiv](https://arxiv.org/abs/2402.19379) ·
Halawi et al. [repo](https://github.com/dannyallover/llm_forecasting) ·
AIA Forecaster [arXiv](https://arxiv.org/abs/2511.07678) ·
Talk Isn't Always Cheap [arXiv](https://arxiv.org/abs/2509.05396) ·
Sycophancy [arXiv](https://arxiv.org/abs/2310.13548) ·
Calibration [arXiv](https://arxiv.org/abs/2306.13063) ·
Prompt sensitivity [arXiv](https://arxiv.org/abs/2310.11324) ·
Non-determinism [arXiv](https://arxiv.org/abs/2408.04667) ·
Model drift [arXiv](https://arxiv.org/abs/2307.09009) ·
FinanceBench [arXiv](https://arxiv.org/abs/2311.11944)

**Real-firm practice and judgment research.**
Lorenz et al. [PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC3107299) ·
Navajas et al. [NHB](https://ideas.repec.org/a/nat/nathum/v2y2018i2d10.1038_s41562-017-0273-4.html) ·
Atanasov et al. [MS](https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2374) ·
Satopää et al. BIN [MS](https://ideas.repec.org/a/inm/ormnsc/v67y2021i12p7599-7618.html) ·
Mandel et al. [JDM](https://www.cambridge.org/core/journals/judgment-and-decision-making/article/boosting-intelligence-analysts-judgment-accuracy-what-works-what-fails/1530E9DAE8F42B2E8FC2B5FE374DB50F) ·
MAP [MIT SMR](https://sloanreview.mit.edu/?p=65293) ·
Scherer [CFA RF](https://rpc.cfainstitute.org/research/foundation/2026/investment-committees-governance-design-choices) ·
Malenko et al. [HBS WP](https://ideas.repec.org/p/hbs/wpaper/21-131.html) ·
Nemeth et al. [EJSP](https://zendy.io/title/10.1002/ejsp.58) ·
Klein pre-mortem [HBR](https://hbr.org/2007/09/performing-a-project-premortem) ·
Best Ideas [PDF](https://personal.lse.ac.uk/polk/research/BestIdeas.pdf) ·
Selling Fast and Buying Slow [NBER](https://nber.org/papers/w29076) ·
Transfer coefficient [FAJ](https://ideas.repec.org/a/taf/ufajxx/v58y2002i5p48-66.html) ·
Lovallo & Sibony [McKinsey](https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-case-for-behavioral-strategy) ·
Harvey et al. [SSRN](https://papers.ssrn.com/abstract=2880641) ·
Bridgewater Principles [site](https://www.principles.com/principles/83b7a067-cd3b-4e57-aaf0-8379ac81b1cb/) ·
Man Group AlphaGPT [Bloomberg](https://www.bloomberg.com/news/articles/2025-07-10/man-group-says-agentic-ai-is-now-devising-quant-trading-signals) ·
Balyasny [OpenAI](https://openai.com/index/balyasny-asset-management) ·
Millennium pod rules (press) [AOL/BI](https://www.aol.com/strategy-69-billion-hedge-fund-193733084.html) ·
Citadel risk [Risk.net](https://www.risk.net/awards/7955800/hedge-fund-of-the-year-citadel) ·
Kaminski & Lo [MIT](https://dspace.mit.edu/handle/1721.1/114876) ·
Man Institute [site](https://www.man.com/maninstitute/youre-fired)

**Portfolio construction and statistics.**
Deflated Sharpe [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) ·
PBO [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) ·
Volatility-managed portfolios [NBER](https://www.nber.org/papers/w22208) ·
1/N [RFS](https://doi.org/10.1093/rfs/hhm075) ·
Grossman-Zhou [MF](https://doi.org/10.1111/j.1467-9965.1993.tb00044.x) ·
Gârleanu-Pedersen [JF](https://doi.org/10.1111/jofi.12080) ·
Kelly properties [PDF](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)

**Security.**
TradeTrap [repo](https://github.com/Yanlewen/TradeTrap) ·
FARSIGHT [arXiv](https://arxiv.org/abs/2609.19705) ·
Poisoning Agentic Alpha [arXiv](https://arxiv.org/abs/2608.24069) ·
Adversarial tweets [arXiv](https://arxiv.org/abs/2205.01094) ·
AutoRedTrader [arXiv](https://arxiv.org/abs/2605.09185) ·
TraderBench [arXiv](https://arxiv.org/abs/2603.00285) ·
Indirect injection [arXiv](https://arxiv.org/abs/2302.12173) ·
AgentDojo [arXiv](https://arxiv.org/abs/2406.13352) ·
AgentPoison [arXiv](https://arxiv.org/abs/2407.12784) ·
MINJA [arXiv](https://arxiv.org/abs/2503.03704) ·
PoisonedRAG [arXiv](https://arxiv.org/abs/2402.07867) ·
MCPTox [arXiv](https://arxiv.org/abs/2508.14925) ·
MCP attack vectors [arXiv](https://arxiv.org/abs/2506.02040) ·
MCP injection experiments [repo](https://github.com/invariantlabs-ai/mcp-injection-experiments) ·
CaMeL [arXiv](https://arxiv.org/abs/2503.18813) ·
Design patterns [arXiv](https://arxiv.org/abs/2506.08837) ·
Spotlighting [arXiv](https://arxiv.org/abs/2403.14720) ·
Adaptive attacks [arXiv](https://arxiv.org/abs/2510.09023) ·
LivePI [arXiv](https://arxiv.org/abs/2605.17986) ·
Prompt Infection [arXiv](https://arxiv.org/abs/2410.07283) ·
Agent-in-the-Middle [arXiv](https://arxiv.org/abs/2502.14847) ·
Debate attack [arXiv](https://arxiv.org/abs/2406.14711) ·
OWASP LLM01 [GitHub](https://github.com/OWASP/www-project-top-10-for-large-language-model-applications/blob/main/2_0_vulns/LLM01_PromptInjection.md)

**Data.**
SEC EDGAR APIs [SEC](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) ·
edgartools [repo](https://github.com/dgunning/edgartools) ·
Massive (Polygon) [AI skills](https://github.com/massive-com/ai), [MCP](https://github.com/massive-com/mcp_massive) ·
yfinance [repo](https://github.com/ranaroussi/yfinance) ·
Alpha Vantage MCP [repo](https://github.com/alphavantage/alpha_vantage_mcp) ·
Financial Datasets MCP [repo](https://github.com/financial-datasets/mcp-server) ·
OpenBB [repo](https://github.com/OpenBB-finance/OpenBB) ·
FRED observations [docs](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) ·
GDELT client [repo](https://github.com/alex9smith/gdelt-doc-api) ·
S&P 500 history [repo](https://github.com/fja05680/sp500) ·
Alpaca [alpaca-py](https://github.com/alpacahq/alpaca-py), [archived docs](https://github.com/alpacahq/alpaca-docs/blob/master/content/trading/paper-trading.md) ·
ib_async [repo](https://github.com/ib-api-reloaded/ib_async) ·
schwab-py [auth docs](https://github.com/alexgolec/schwab-py/blob/main/docs/auth.rst)

**Legal, regulatory and operations.**
Advisers Act §202 [LII](https://www.law.cornell.edu/uscode/text/15/80b-2) ·
Rule 203A-1 [LII](https://www.law.cornell.edu/cfr/text/17/275.203A-1) ·
Reg D 506 [LII](https://www.law.cornell.edu/cfr/text/17/230.506) ·
Marketing Rule [LII](https://www.law.cornell.edu/cfr/text/17/275.206(4)-1) ·
Books and records [LII](https://www.law.cornell.edu/cfr/text/17/275.204-2) ·
Qualified-client order IA-6961 [SEC](https://www.sec.gov/files/rules/ia/2026/ia-6961.pdf) ·
FINRA SRO filings [SEC](https://www.sec.gov/rules-regulations/self-regulatory-organization-rulemaking/finra) ·
FINRA 4210 [FINRA](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210) ·
FINRA RN 24-09 [FINRA](https://www.finra.org/rules-guidance/notices/24-09) ·
CFTC 4.13 [LII](https://www.law.cornell.edu/cfr/text/17/4.13) ·
Wash sales [LII](https://www.law.cornell.edu/uscode/text/26/1091) ·
Rule 15c3-5 [LII](https://www.law.cornell.edu/cfr/text/17/240.15c3-5) ·
Knight Capital [Chicago Fed](https://www.chicagofed.org/-/media/publications/chicago-fed-letter/2012/cfloctober2012-303-pdf.pdf) ·
SR 11-7 [Fed](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) ·
NIST AI 600-1 [PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) ·
Anthropic pricing [docs](https://platform.claude.com/docs/en/about-claude/pricing) ·
Anthropic Usage Policy [site](https://www.anthropic.com/legal/aup)
