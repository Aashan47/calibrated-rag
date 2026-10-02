"""Uncertainty quantification for the eval metrics (stdlib only).

A single point estimate ("83.1% accuracy") hides sampling noise on a 300-question slice. We
report **95% bootstrap confidence intervals** so the numbers are honest about their precision,
and a one-sided check for whether the conformal guarantee holds with margin. This is the kind
of rigor a production eval needs before anyone trusts the headline number.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence


def bootstrap_ci(data: Sequence, statistic: Callable[[Sequence], float],
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for an arbitrary statistic over `data` (a sequence of items).

    Returns {'point', 'lo', 'hi', 'se', 'n'}. Resamples the items with replacement n_boot
    times, recomputes the statistic, and takes the empirical [alpha/2, 1-alpha/2] quantiles.
    """
    data = list(data)
    n = len(data)
    point = float(statistic(data)) if n else 0.0
    if n < 2:
        return {"point": round(point, 4), "lo": round(point, 4), "hi": round(point, 4),
                "se": 0.0, "n": n}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        boots.append(float(statistic(sample)))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mean = sum(boots) / len(boots)
    se = (sum((b - mean) ** 2 for b in boots) / (len(boots) - 1)) ** 0.5
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "se": round(se, 4), "n": n}


def proportion(flags: Sequence[bool]) -> float:
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def fmt_pct(ci: dict) -> str:
    """'83.1% [78.0, 88.2]' — point with 95% CI, both as percentages."""
    return f"{ci['point']*100:.1f}% [{ci['lo']*100:.1f}, {ci['hi']*100:.1f}]"
