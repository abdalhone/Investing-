"""Honest evaluation (P10).

* :mod:`.metrics`: performance metrics with significance corrections (PSR, DSR, MinTRL),
  Newey-West t-stats and HAC regression.
* :mod:`.engine`: the walk-forward backtest with baselines, a buffered model-cutoff split,
  and re-execution of recorded decisions at other costs.
* :mod:`.evaluation`: "is there skill, and would we know?", assembled from :mod:`.ic`
  (per-date rank IC, paired IC tests, power), :mod:`.random_books` (composition-matched
  null), cost sensitivity with a breakeven bisection, and :mod:`.factors` (ETF factor
  attribution).
"""
