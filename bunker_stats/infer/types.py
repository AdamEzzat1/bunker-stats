"""Rich result objects for the inference module.

Each class is a small dataclass over the raw dict a Rust kernel returns, plus a
little input-derived metadata (sample sizes, chosen ``alternative``, effect
size where cheap). They all follow the house protocol from
:mod:`bunker_stats._result`:

* tuple unpacking (``statistic, pvalue = result``) and integer indexing,
* ``.to_dict()`` (populated fields only; NumPy scalars -> Python scalars),
* ``.info()`` (a readable multi-line summary), and
* ``.conclusion(alpha=0.05)`` for the hypothesis tests.

These are only ever returned when a facade is called with ``rich=True``; the
default dict return is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .._result import HypothesisResult

__all__ = [
    "TTestResult",
    "ChiSquareResult",
    "MannWhitneyResult",
    "KSResult",
    "ANOVAResult",
    "CorrelationTestResult",
    "NormalityResult",
]


@dataclass
class TTestResult(HypothesisResult):
    """Result of a one-sample, two-sample or paired t-test.

    Unpacks as ``statistic, pvalue``. ``mean``/``mean_x``/``mean_y`` and
    ``equal_var`` are populated according to which test produced it.
    """

    statistic: float
    pvalue: float
    df: float
    alternative: str = "two-sided"
    kind: str = "two-sample"
    mean: Optional[float] = None
    mean_x: Optional[float] = None
    mean_y: Optional[float] = None
    equal_var: Optional[bool] = None
    effect_size: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n1: Optional[int] = None
    n2: Optional[int] = None

    _title = "t-Test"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("Test:", self.kind),
            ("Alternative:", self.alternative),
            ("df:", self.df),
        ]
        if self.mean is not None:
            rows.append(("Mean:", self.mean))
        if self.mean_x is not None:
            rows.append(("Mean x / y:", f"{self.mean_x:.6g} / {self.mean_y:.6g}"))
        if self.equal_var is not None:
            rows.append(("Equal variance:", self.equal_var))
        if self.effect_size is not None:
            rows.append(("Cohen's d:", self.effect_size))
        if self.n2 is not None:
            rows.append(("n1 / n2:", f"{self.n1} / {self.n2}"))
        elif self.n1 is not None:
            rows.append(("n:", self.n1))
        return rows


@dataclass
class ChiSquareResult(HypothesisResult):
    """Result of a chi-square goodness-of-fit or independence test.

    Unpacks as ``statistic, pvalue``. ``observed`` echoes the input table;
    ``expected`` is populated only when the kernel provides it (currently None).
    """

    statistic: float
    pvalue: float
    dof: float
    test_type: str = "goodness_of_fit"
    observed: Optional[np.ndarray] = None
    expected: Optional[np.ndarray] = None

    _title = "Chi-Square Test"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        return [
            ("Test type:", self.test_type),
            ("Degrees of freedom:", self.dof),
        ]


@dataclass
class MannWhitneyResult(HypothesisResult):
    """Result of the Mann-Whitney U test (Wilcoxon rank-sum).

    Unpacks as ``statistic, pvalue``. ``rank_biserial`` is the rank-biserial
    effect size, computed from the same inputs.
    """

    statistic: float
    pvalue: float
    alternative: str = "two-sided"
    rank_biserial: Optional[float] = None
    n1: Optional[int] = None
    n2: Optional[int] = None

    _title = "Mann-Whitney U Test"
    _h0 = "H0 (the two distributions are equal)"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [("Alternative:", self.alternative)]
        if self.rank_biserial is not None:
            rows.append(("Rank-biserial:", self.rank_biserial))
        if self.n1 is not None:
            rows.append(("n1 / n2:", f"{self.n1} / {self.n2}"))
        return rows


@dataclass
class KSResult(HypothesisResult):
    """Result of the one-sample Kolmogorov-Smirnov test.

    Unpacks as ``statistic, pvalue``. ``distribution`` is the reference CDF name
    that was tested against.
    """

    statistic: float
    pvalue: float
    alternative: str = "two-sided"
    distribution: Optional[str] = None
    n: Optional[int] = None

    _title = "Kolmogorov-Smirnov Test (one-sample)"
    _h0 = "H0 (the sample follows the reference distribution)"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [("Alternative:", self.alternative)]
        if self.distribution is not None:
            rows.append(("Distribution:", self.distribution))
        if self.n is not None:
            rows.append(("n:", self.n))
        return rows


@dataclass
class ANOVAResult(HypothesisResult):
    """Result of one-way ANOVA (the omnibus F-test).

    Unpacks as ``statistic, pvalue``. ``eta_squared``/``omega_squared`` are the
    effect sizes, computed from the same groups.
    """

    statistic: float
    pvalue: float
    df_between: float
    df_within: float
    eta_squared: Optional[float] = None
    omega_squared: Optional[float] = None
    n_groups: Optional[int] = None
    n_total: Optional[int] = None

    _title = "One-Way ANOVA (F-test)"
    _h0 = "H0 (all group means are equal)"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("df (between):", self.df_between),
            ("df (within):", self.df_within),
        ]
        if self.eta_squared is not None:
            rows.append(("Eta-squared:", self.eta_squared))
        if self.omega_squared is not None:
            rows.append(("Omega-squared:", self.omega_squared))
        if self.n_groups is not None:
            rows.append(("Groups / N:", f"{self.n_groups} / {self.n_total}"))
        return rows


@dataclass
class CorrelationTestResult(HypothesisResult):
    """Result of a Pearson or Spearman correlation test.

    Unpacks as ``r, pvalue`` -- the correlation coefficient is the headline
    quantity. ``statistic`` is the underlying t-statistic. ``ci_low``/``ci_high``
    are populated for the Pearson method only.
    """

    r: float
    statistic: float
    pvalue: float
    df: float
    method: str = "pearson"
    n: Optional[int] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    _fields = ("r", "pvalue")
    _title = "Correlation Test"
    _h0 = "H0 (no correlation, rho = 0)"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("Correlation r:", self.r),
            ("Method:", self.method),
            ("df:", self.df),
        ]
        if self.n is not None:
            rows.append(("n:", self.n))
        if self.ci_low is not None:
            rows.append(("95% CI:", f"[{self.ci_low:.4g}, {self.ci_high:.4g}]"))
        return rows


@dataclass
class NormalityResult(HypothesisResult):
    """Result of a normality test (Jarque-Bera or Anderson-Darling).

    Unpacks as ``statistic, pvalue``. Anderson-Darling has no p-value in the
    kernel, so ``pvalue`` is ``None`` there and the verdict falls back to the
    5% critical value (A* > 0.787 rejects normality).
    """

    statistic: float
    pvalue: Optional[float]
    method: str = "jarque_bera"
    n: Optional[int] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    critical_value_5pct: Optional[float] = None

    _title = "Normality Test"
    _h0 = "H0 (data is normally distributed)"

    def is_normal(self, alpha: float = 0.05) -> bool:
        """Whether normality is *not* rejected at ``alpha``.

        Uses the p-value when available (Jarque-Bera); otherwise compares the
        statistic to the 5% critical value (Anderson-Darling), in which case the
        ``alpha`` argument is ignored.
        """
        if self.pvalue is not None and not math.isnan(self.pvalue):
            return self.pvalue >= alpha
        if self.critical_value_5pct is not None:
            return self.statistic <= self.critical_value_5pct
        return False

    def conclusion(self, alpha: float = 0.05) -> str:
        if self.pvalue is not None and not math.isnan(self.pvalue):
            return super().conclusion(alpha)
        # Anderson-Darling: decide via the 5% critical value.
        if self.critical_value_5pct is None:
            return "Inconclusive: no p-value or critical value available"
        if self.statistic > self.critical_value_5pct:
            return (
                f"Reject {self._h0} at the 5% level "
                f"(A*={self.statistic:.4g} > {self.critical_value_5pct:g})"
            )
        return (
            f"Fail to reject {self._h0} at the 5% level "
            f"(A*={self.statistic:.4g} <= {self.critical_value_5pct:g})"
        )

    def _detail_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [("Method:", self.method)]
        if self.skewness is not None:
            rows.append(("Skewness:", self.skewness))
        if self.kurtosis is not None:
            rows.append(("Kurtosis:", self.kurtosis))
        if self.critical_value_5pct is not None:
            rows.append(("5% critical value:", self.critical_value_5pct))
        if self.n is not None:
            rows.append(("n:", self.n))
        return rows
