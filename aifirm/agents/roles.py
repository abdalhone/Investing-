"""Analyst role specifications: what each analyst judges and which slice of facts it sees.

Roles are defined by *information*, not personality (P5). Each analyst reads a different
slice of the fact sheet, so the committee pools genuinely different evidence instead of
several restatements of the same evidence (the "hidden profile" problem: groups that
share information converge on it and miss what only one member knows). There are no
famous-investor personas: persona prompting has not been shown to improve accuracy, and
it invites the model to role-play a style instead of reading the data.

The checklists name concrete fact keys (e.g. ``mom_12_1``) so the analyst knows which
lines of its slice answer which question. They describe what to evaluate, never what to
conclude.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalystRole:
    name: str                    # "fundamental" | "technical" | "macro" | "events"
    title: str
    mandate: str                 # what this analyst is responsible for judging
    checklist: tuple[str, ...]   # concrete things to evaluate (task-specific, not persona)
    sections: tuple[str, ...]    # FactSheet.render sections this analyst sees (its data slice)
    extra_fact_keys: tuple[str, ...] = ()  # ticker facts from other sections it may also see


FUNDAMENTAL = AnalystRole(
    name="fundamental",
    title="Fundamental analyst",
    mandate=(
        "Judge whether the company's financial trajectory and its valuation, as of the latest SEC filings, make the "
        "stock more or less likely than the benchmark to outperform over the horizon. Your slice is the as-filed "
        "fundamentals and valuation multiples, plus the last price and the 12-month return so you can tell whether the "
        "story is already reflected in the price. Over a short horizon, fundamentals matter mainly through a mismatch "
        "between business quality or trajectory and what the valuation already assumes; filed numbers are public, so "
        "the question is whether the market is mispricing them, not whether they are good."
    ),
    checklist=(
        "Growth quality: revenue growth (revenue_growth_yoy) and whether net-income and operating-cash-flow growth "
        "(net_income_growth_yoy, op_cash_flow_growth_yoy) confirm it or diverge from it.",
        "Margins: level of gross, operating and net margins (gross_profit_margin, operating_income_margin, "
        "net_income_margin), and whether net income growing faster or slower than revenue implies margins are "
        "expanding or compressing. Only TTM levels are shown; do not claim a margin trend the facts cannot show.",
        "Cash conversion: free cash flow (fcf, op_cash_flow, capex) versus net income. FCF well below net income "
        "suggests low-quality, accrual-driven earnings; FCF above net income is a quality signal. Note stock-based "
        "compensation (sbc) where it is large relative to FCF.",
        "Balance-sheet risk: leverage (debt_to_equity, long_term_debt versus cash), liquidity (current_ratio), and "
        "total liabilities versus total assets.",
        "Dilution: share_count_change_yoy (positive = dilution, negative = buybacks) and stock-based compensation.",
        "Valuation versus growth: pe_ttm, ps_ttm, pb, earnings_yield and fcf_yield judged against growth and margins. "
        "Expensive and decelerating is the riskiest combination; cheap and improving is the most favorable. You have no "
        "peer or history data, so do not claim the stock is cheap or expensive versus peers or its own past.",
        "Staleness and pricing: check the as-of dates (TTM figures can be months old) and compare the story with ret_12m. "
        "A strong 12-month return means good fundamentals are probably already priced.",
        "Missing data: if revenue, earnings or shares outstanding are listed under DATA GAPS, valuation and quality "
        "cannot be judged; abstain rather than reason from memory.",
    ),
    sections=("fund",),
    extra_fact_keys=("price", "ret_12m"),
)

TECHNICAL = AnalystRole(
    name="technical",
    title="Technical analyst",
    mandate=(
        "Judge what the stock's own price history (trend, momentum, short-term extension, volatility and relative "
        "strength) implies about its odds of beating the benchmark over the horizon. Your slice is the price, trend and "
        "risk facts only. Rely on effects with documented evidence and modest size: intermediate-term (12-1 month) "
        "momentum tends to persist; returns over the most recent month tend to partially reverse; stocks far above their "
        "averages or with very high RSI are more exposed to pullbacks; high volatility widens the range of outcomes "
        "without changing the expected direction much."
    ),
    checklist=(
        "Trend: sma50_sma200 (above 1 = uptrend) and price_vs_sma200 (how far the price is from its 200-day average).",
        "Momentum 12-1 (mom_12_1): the most robust price signal at this horizon. Positive momentum tilts the odds "
        "modestly in favor; negative momentum modestly against.",
        "Mean-reversion risk: ret_1m (a very strong or very weak latest month tends to partially reverse), rsi_14 "
        "(above 70 overbought, below 30 oversold) and dist_52w_high (at the highs versus deep below them).",
        "Relative strength: rel_ret_3m and rel_ret_12m versus the benchmark. Is the stock leading or lagging, and is that "
        "consistent with the 12-1 momentum?",
        "Volatility regime: vol_20d versus vol_60d (short-term volatility rising above the longer window signals stress or "
        "a regime change) and max_dd_1y. High or rising volatility should widen your scenarios and pull your probability "
        "toward 0.50.",
        "Market sensitivity: beta_1y. With a high beta, outperformance depends heavily on the market's direction, which "
        "your slice cannot forecast; this argues for a probability closer to 0.50.",
        "Liquidity: adv_20d. Very thin trading makes price signals noisier.",
        "Conflicts: when signals disagree (e.g. strong momentum but overbought), say so and keep the probability close to "
        "0.50 rather than picking the more exciting signal.",
    ),
    sections=("price",),
)

MACRO = AnalystRole(
    name="macro",
    title="Macro and market-regime analyst",
    mandate=(
        "Judge how the current macro and market regime is likely to affect this particular stock relative to the "
        "benchmark, given its market sensitivity (beta_1y), volatility (vol_60d), sector and recent relative performance. "
        "Your slice is the macro and market facts plus those four stock facts and the sector. You are not forecasting "
        "the market: the question is relative performance, so the regime matters only through how this stock's beta and "
        "sector exposure interact with it. Macro conditions move slowly and are widely watched, so over one month the "
        "honest relative edge from macro is usually small."
    ),
    checklist=(
        "Rates: level and 3-month change of the 10-year (MACRO.DGS10), 2-year (MACRO.DGS2), 3-month bill (MACRO.DGS3MO) "
        "and fed funds (MACRO.DFF). Rising long rates tend to hurt long-duration (high-valuation, growth) equities and "
        "help some financials; falling rates the reverse.",
        "Yield curve: MACRO.curve_2s10s. Inversion and the subsequent re-steepening have historically been late-cycle "
        "signals that favor defensive over cyclical exposure.",
        "Credit: high-yield spread (MACRO.BAMLH0A0HYM2) level and 3-month change. Widening spreads mean risk-off, which "
        "tends to hurt high-beta and leveraged companies relative to the market.",
        "Volatility: VIX (MACRO.VIXCLS) level and change, and the benchmark's realized volatility (MKT.*_vol_20d). "
        "Elevated or rising volatility tends to favor low-beta names; calm, falling volatility favors high-beta names.",
        "Inflation and labor: CPI inflation (MACRO.CPIAUCSL) and unemployment (MACRO.UNRATE) as indicators of policy "
        "pressure. Note their publication lags (as-of dates).",
        "Market trend: the benchmark versus its 200-day average (MKT.*_vs_sma200) and its 3- and 12-month returns.",
        "Interaction: combine the regime with this stock's beta_1y, vol_60d, sector and rel_ret_3m. A beta near 1 means "
        "the regime implies little relative edge, so stay close to 0.50. State explicitly which way beta times regime "
        "points and how strongly.",
        "Missing data: if the macro section is missing or listed under DATA GAPS, abstain.",
    ),
    sections=("macro",),
    extra_fact_keys=("beta_1y", "vol_60d", "ret_3m", "rel_ret_3m"),
)

EVENTS = AnalystRole(
    name="events",
    title="Events and filings analyst",
    mandate=(
        "Judge what recent corporate events (SEC filings and, when available, headlines) imply for the stock's odds of "
        "beating the benchmark over the horizon, and how much remains unknown. Your slice is the dated filing events and "
        "headlines plus the last price, the 1-month return and 20-day volatility, so you can see how the market has "
        "reacted so far. Separate genuinely new information that may not be fully priced from routine disclosures. "
        "Most windows contain only routine filings; then the honest answer is close to 0.50."
    ),
    checklist=(
        "8-K items: which items were filed and what they usually imply. An earnings release (results of operations) is "
        "new information; ret_1m shows the market's reaction so far. Non-reliance on prior financials (restatement), a "
        "change of auditor, a delisting notice, bankruptcy, a material impairment, and exit or restructuring costs are "
        "usually negative. Material agreements and acquisitions are ambiguous without details. Officer or director "
        "changes depend on who and why, which you cannot see.",
        "Dilution signals: unregistered equity sales and registration statements (S-1, S-3) can precede new share issuance.",
        "Filing cadence: are periodic reports (10-Q, 10-K) present as expected? A missing periodic report is a warning "
        "only if the window is long enough to expect one; say how confident you are.",
        "Insider filings: Form 4 counts show activity, not direction. The fact sheet says the direction is not parsed; do "
        "not infer buying or selling from a count.",
        "Headlines: if present, separate reported facts from opinion. Analyst ratings, price targets and commentary are "
        "not evidence about the business. If no headlines are available, do not assume any news.",
        "Post-event drift: after earnings news, prices tend to keep drifting in the direction of the initial reaction for "
        "weeks. You cannot see the surprise itself; ret_1m after an earnings 8-K is only a rough proxy, so weight it lightly.",
        "Volatility: vol_20d. Elevated short-term volatility around events widens the range of outcomes.",
        "Unknowns: list what you cannot see (filing contents, the surprise versus expectations, insider direction). The "
        "more that is unknown, the closer to 0.50 you should be.",
    ),
    sections=("events", "news"),
    extra_fact_keys=("ret_1m", "price", "vol_20d"),
)

ROLES: dict[str, AnalystRole] = {r.name: r for r in (FUNDAMENTAL, TECHNICAL, MACRO, EVENTS)}
