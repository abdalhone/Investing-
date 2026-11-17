"""Execution: a paper broker with an honest cost model, and a guarded Alpaca adapter.

Why the cost model matters (P10): many published LLM-trading results disappear once
fills happen at the next open and pay commissions, spread and market impact (FINSABER;
Bailey and Lopez de Prado). Every simulated trade here pays:

* commission: ``commission_bps`` of notional;
* slippage: ``slippage_bps`` plus a square-root impact term,
  ``impact_coefficient * 1e4 * daily_vol * sqrt(|notional| / ADV)`` bps (the "square-root
  law" of market impact; Almgren et al. 2005, Toth et al. 2011).

Buys fill *above* the reference price and sells *below* it, so a round trip at an
unchanged price always loses money. Cash can never go negative: a buy that cannot be paid
for is scaled down to the cash available. Fractional shares are allowed.

The Alpaca adapter talks to the documented Alpaca Trading API v2 over HTTPS. Because it
can move real money, it refuses to start without keys, refuses a live endpoint unless
``allow_live=True``, and with ``require_approval`` refuses to submit unless the caller
passes ``approve=True`` (a human said yes). Moving from paper to real money is a human
decision (ARCHITECTURE.md section 4).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests

from ..config import CostConfig
from ..schemas import Fill, Order

ZERO_SHARES = 1e-12   # share counts below this are floating-point residue


@runtime_checkable
class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...       # shares
    def cash(self) -> float: ...
    def nav(self, prices: dict[str, float]) -> float: ...
    def submit(self, orders: list[Order], fill_prices: dict[str, float], date: str,
               adv: dict[str, float] | None = None, daily_vol: dict[str, float] | None = None) -> list[Fill]: ...


def _positive(x: object) -> float | None:
    """``float(x)`` if it is a finite positive number, else None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def execution_cost(notional: float, price: float, adv: float | None, daily_vol: float | None,
                   cfg: CostConfig) -> tuple[float, float]:
    """``(commission, slippage_cost)`` in dollars for a trade of ``|notional|`` at ``price``.

    ``daily_vol`` is the daily return standard deviation as a fraction (e.g. 0.02) and
    ``adv`` the average daily dollar volume. Without both, only the flat ``slippage_bps``
    applies. ``price`` must be a real, positive price: a trade at any other price is a bug
    upstream, so it raises rather than quietly costing nothing.
    """
    if _positive(price) is None:
        raise ValueError(f"execution price must be finite and positive, got {price!r}")
    n = abs(float(notional))
    if not math.isfinite(n):
        raise ValueError(f"notional must be finite, got {notional!r}")
    bps = cfg.slippage_bps
    a, dv = _positive(adv), _positive(daily_vol)
    if a is not None and dv is not None:
        bps += cfg.impact_coefficient * 1e4 * dv * math.sqrt(n / a)
    return n * cfg.commission_bps / 1e4, n * bps / 1e4


class PaperBroker:
    """Simulated broker. ``Fill.price`` is the execution price *including* slippage
    (``slippage_cost`` is reported for attribution and is already inside ``price``);
    commission is charged to cash on top. Hence, for every fill:

    * buy:  cash -= quantity * price + commission
    * sell: cash += quantity * price - commission

    and ``nav(prices) == cash + sum(shares * prices)`` always holds.

    Sells are processed before buys so their proceeds can fund the buys. Short sales are
    refused unless ``allow_short=True`` (sells are capped at the shares held). With a
    ``state_path`` the state is saved after every ``submit``; call ``load()`` to resume.
    """

    def __init__(self, cash: float, cfg: CostConfig, state_path: str | None = None, *,
                 allow_short: bool = False):
        start = float(cash)
        if not math.isfinite(start) or start < 0:
            raise ValueError(f"starting cash must be finite and >= 0, got {cash!r}")
        self.cfg = cfg
        self.state_path = state_path
        self.allow_short = allow_short
        self._cash = start
        self._positions: dict[str, float] = {}
        self._marks: dict[str, float] = {}       # last known price per ticker (stale-mark fallback)
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.last_rejections: list[str] = []     # why orders in the last submit() did not (fully) fill

    # -- Broker protocol -------------------------------------------------------------
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def nav(self, prices: dict[str, float]) -> float:
        """Cash plus shares marked at ``prices``. A ticker without a valid price is marked at
        its last known price (a stale mark beats a NAV that silently drops a holding); with
        no price ever seen, the NAV is unknown and this raises."""
        total = self._cash
        for t, q in sorted(self._positions.items()):
            p = _positive(prices.get(t))
            if p is not None:
                self._marks[t] = p
            elif t in self._marks:
                p = self._marks[t]
            else:
                raise ValueError(f"cannot mark {t}: no price given and none seen before")
            total += q * p
        return total

    def submit(self, orders: list[Order], fill_prices: dict[str, float], date: str,
               adv: dict[str, float] | None = None, daily_vol: dict[str, float] | None = None) -> list[Fill]:
        """Fill ``orders`` at ``fill_prices`` (in backtests: the next open, P3/P10)."""
        adv, daily_vol = adv or {}, daily_vol or {}
        self.last_rejections = []
        fills = []
        for o in [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]:
            price = _positive(fill_prices.get(o.ticker))
            qty = _positive(o.quantity)
            if price is None or qty is None:
                self.last_rejections.append(f"{o.ticker} {o.side}: no valid fill price or quantity")
                continue
            fill = (self._sell if o.side == "sell" else self._buy)(
                o.ticker, qty, price, date, adv.get(o.ticker), daily_vol.get(o.ticker))
            if fill is not None:
                fills.append(fill)
                self._marks[o.ticker] = price
        if self.state_path:
            self.save()
        return fills

    # -- internals -------------------------------------------------------------------
    def _sell(self, t: str, qty: float, price: float, date: str, adv: float | None,
              daily_vol: float | None) -> Fill | None:
        held = self._positions.get(t, 0.0)
        if not self.allow_short and qty > held:
            self.last_rejections.append(f"{t} sell: {qty:.6g} requested, {max(held, 0.0):.6g} held (no shorting)")
            qty = max(held, 0.0)
        if qty <= 0:
            return None
        commission, slippage = execution_cost(qty * price, price, adv, daily_vol, self.cfg)
        proceeds = qty * price - slippage - commission
        if self._cash + proceeds < 0:
            self.last_rejections.append(f"{t} sell: costs exceed proceeds and cash")
            return None
        return self._book(t, -qty, price - slippage / qty, commission, slippage, date, "sell", proceeds)

    def _buy(self, t: str, qty: float, price: float, date: str, adv: float | None,
             daily_vol: float | None) -> Fill | None:
        commission, slippage = execution_cost(qty * price, price, adv, daily_vol, self.cfg)
        total = qty * price + slippage + commission
        if total > self._cash:
            if self._cash <= 0:
                self.last_rejections.append(f"{t} buy: no cash")
                return None
            # Costs grow at least proportionally with size, so this rescale is affordable;
            # the tiny haircut absorbs floating-point error.
            requested = qty
            qty = qty * self._cash / total * (1 - 1e-12)
            commission, slippage = execution_cost(qty * price, price, adv, daily_vol, self.cfg)
            total = qty * price + slippage + commission
            self.last_rejections.append(f"{t} buy: scaled from {requested:.6g} to {qty:.6g} shares (cash)")
        if qty <= 0 or total > self._cash:
            self.last_rejections.append(f"{t} buy: unaffordable")
            return None
        return self._book(t, qty, price + slippage / qty, commission, slippage, date, "buy", -total)

    def _book(self, t: str, signed_qty: float, exec_price: float, commission: float, slippage: float,
              date: str, side: str, cash_change: float) -> Fill:
        self._cash = self._cash + cash_change
        shares = self._positions.get(t, 0.0) + signed_qty
        if abs(shares) < ZERO_SHARES:
            self._positions.pop(t, None)
        else:
            self._positions[t] = shares
        self.total_commission += commission
        self.total_slippage += slippage
        return Fill(ticker=t, side=side, quantity=abs(signed_qty), price=exec_price,
                    commission=commission, slippage_cost=slippage, date=date)

    # -- persistence -----------------------------------------------------------------
    def save(self, path: str | None = None) -> None:
        """Write the state as JSON, atomically (write a temp file, then rename)."""
        target = path or self.state_path
        if not target:
            raise ValueError("no state_path to save to")
        data = {"cash": self._cash, "positions": self._positions, "marks": self._marks,
                "total_commission": self.total_commission, "total_slippage": self.total_slippage}
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, p)

    def load(self, path: str | None = None) -> bool:
        """Restore state saved by :meth:`save`. Returns False (state unchanged) if there is no
        file yet; raises on a corrupt file, because trading from a guessed state is worse
        than not trading."""
        target = path or self.state_path
        if not target or not Path(target).exists():
            return False
        data = json.loads(Path(target).read_text())
        cash = float(data["cash"])
        positions = {str(k): float(v) for k, v in data.get("positions", {}).items()}
        marks = {str(k): float(v) for k, v in data.get("marks", {}).items()}
        numbers = [cash, *positions.values(), *marks.values()]
        if cash < 0 or not all(math.isfinite(x) for x in numbers):
            raise ValueError(f"corrupt paper-broker state in {target}")
        self._cash, self._positions, self._marks = cash, positions, marks
        self.total_commission = float(data.get("total_commission", 0.0))
        self.total_slippage = float(data.get("total_slippage", 0.0))
        return True


# --------------------------------------------------------------------------- Alpaca

PAPER_URL = "https://paper-api.alpaca.markets"


class ApprovalRequired(PermissionError):
    """Raised when orders would reach a real broker without explicit human approval."""


class AlpacaBroker:
    """Minimal Alpaca Trading API v2 client: account, positions, market orders.

    Configuration (environment): ``ALPACA_API_KEY_ID``, ``ALPACA_API_SECRET_KEY`` and
    optionally ``ALPACA_BASE_URL`` (default: the paper endpoint). Guards, all fail-closed:

    * no keys -> refuses to construct;
    * any endpoint other than the paper URL counts as live and needs ``allow_live=True``;
    * with ``require_approval`` (the config default) ``submit`` needs ``approve=True``.

    Orders are fractional-quantity market orders with ``time_in_force="day"``, sells first.
    Each carries a deterministic ``client_order_id`` (date, ticker, side, quantity), so a
    retried submission of the same plan is rejected by Alpaca as a duplicate instead of
    doubling the position. Any HTTP error stops the batch (no partial plan keeps firing).

    Market orders normally fill moments after submission, so ``submit`` returns fills only
    for orders Alpaca already reports as ``filled``; every raw order response is kept in
    ``last_orders`` for reconciliation against ``positions()`` before the next cycle.
    Untested against the live service in this repository (no keys); the request formats
    follow Alpaca's public v2 documentation.
    """

    def __init__(self, *, require_approval: bool = True, allow_live: bool = False,
                 base_url: str | None = None, session: requests.Session | None = None, timeout: float = 10.0):
        key = os.environ.get("ALPACA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not key or not secret:
            raise RuntimeError("AlpacaBroker refuses to start: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        url = (base_url or os.environ.get("ALPACA_BASE_URL") or PAPER_URL).rstrip("/")
        url = url.removesuffix("/v2")
        if url != PAPER_URL and not allow_live:
            raise PermissionError(f"{url} is not the Alpaca paper endpoint; live trading needs allow_live=True "
                                  f"(a human decision)")
        self.base_url = url
        self.require_approval = require_approval
        self.timeout = timeout
        self.session = session or requests.Session()
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.last_orders: list[dict[str, Any]] = []

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self.session.request(method, self.base_url + path, headers=self._headers,
                                    timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def account(self) -> dict[str, Any]:
        return self._request("GET", "/v2/account")

    def _raw_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/positions")

    def positions(self) -> dict[str, float]:
        out = {}
        for p in self._raw_positions():
            q = float(p["qty"])
            out[p["symbol"]] = -abs(q) if p.get("side") == "short" else q
        return out

    def cash(self) -> float:
        return float(self.account()["cash"])

    def nav(self, prices: dict[str, float]) -> float:
        """Cash plus positions marked at ``prices``, falling back to Alpaca's ``current_price``."""
        total = self.cash()
        for p in self._raw_positions():
            q = float(p["qty"])
            q = -abs(q) if p.get("side") == "short" else q
            px = _positive(prices.get(p["symbol"])) or float(p["current_price"])
            total += q * px
        return total

    def submit(self, orders: list[Order], fill_prices: dict[str, float], date: str,
               adv: dict[str, float] | None = None, daily_vol: dict[str, float] | None = None,
               *, approve: bool = False) -> list[Fill]:
        """Send market orders. ``fill_prices`` are the reference prices used to report
        slippage on immediate fills; ``adv`` and ``daily_vol`` are accepted for protocol
        compatibility (the venue, not a model, decides real costs)."""
        if self.require_approval and not approve:
            raise ApprovalRequired("orders for a real broker need human approval: call submit(..., approve=True)")
        self.last_orders = []
        fills = []
        for o in [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]:
            if _positive(o.quantity) is None:
                continue
            qty = Decimal(str(o.quantity)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            ident = hashlib.sha256(f"{date}|{o.ticker}|{o.side}|{qty}".encode()).hexdigest()[:24]
            body = {"symbol": o.ticker, "qty": str(qty), "side": o.side, "type": "market",
                    "time_in_force": "day", "client_order_id": f"aifirm-{ident}"}
            resp = self._request("POST", "/v2/orders", json=body)
            self.last_orders.append(resp)
            filled_qty = _positive(resp.get("filled_qty"))
            avg = _positive(resp.get("filled_avg_price"))
            if resp.get("status") == "filled" and filled_qty and avg:
                ref = _positive(fill_prices.get(o.ticker))
                # Signed implementation shortfall vs the reference: positive = paid up / sold down.
                slippage = 0.0
                if ref is not None:
                    slippage = ((avg - ref) if o.side == "buy" else (ref - avg)) * filled_qty
                fills.append(Fill(ticker=o.ticker, side=o.side, quantity=filled_qty, price=avg,
                                  commission=0.0, slippage_cost=slippage, date=date))
        return fills

