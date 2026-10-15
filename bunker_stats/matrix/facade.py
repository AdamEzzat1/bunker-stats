"""``rich=True`` facades for the matrix estimators."""
from __future__ import annotations

import numpy as np

from .._result import rich_facade
from .types import CorrelationMatrixResult, CovarianceMatrixResult


def _n_obs(args):
    """Row count of the input matrix, if we can see it."""
    if not args:
        return None
    try:
        return int(np.asarray(args[0]).shape[0])
    except Exception:
        return None


def _build_corr(matrix, args, kwargs):
    return CorrelationMatrixResult(
        matrix=np.asarray(matrix),
        columns=kwargs.get("columns"),
        method=kwargs.get("method", "pearson"),
        n_obs=_n_obs(args),
    )


def _build_cov(matrix, args, kwargs):
    return CovarianceMatrixResult(
        matrix=np.asarray(matrix),
        columns=kwargs.get("columns"),
        ddof=kwargs.get("ddof", 1),
        n_obs=_n_obs(args),
    )


def wrap_matrix(raw_funcs: dict):
    """Return rich-wrapped ``corr_matrix`` / ``cov_matrix`` (when present)."""
    wrapped = {}
    if raw_funcs.get("corr_matrix") is not None:
        wrapped["corr_matrix"] = rich_facade(
            raw_funcs["corr_matrix"], _build_corr, name="corr_matrix",
            builder_kwargs=("columns", "method"),
        )
    if raw_funcs.get("cov_matrix") is not None:
        wrapped["cov_matrix"] = rich_facade(
            raw_funcs["cov_matrix"], _build_cov, name="cov_matrix",
            builder_kwargs=("columns", "ddof"),
        )
    return wrapped
