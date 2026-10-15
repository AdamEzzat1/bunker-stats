"""Shared base for rich result objects across bunker-stats modules.

The time-series module established a house style for result objects: tuple
unpacking, integer indexing, ``.to_dict()``, ``.info()`` and (for hypothesis
tests) ``.conclusion()``. This module factors that protocol into two small
mixins so every module's ``types.py`` can offer the same ergonomics without
re-implementing the plumbing:

* :class:`RichResult` -- iteration, indexing, ``to_dict`` and ``__repr__`` for
  any dataclass result.
* :class:`HypothesisResult` -- adds ``conclusion(alpha)`` / ``is_significant``
  and a default ``info()`` for anything with a ``statistic`` and ``pvalue``.

Design rules
------------
* Rich objects are always **opt-in** (``rich=True``); default returns are never
  changed. See :func:`rich_facade`.
* Iteration/indexing yield only the *primary* fields named in ``_fields`` (e.g.
  ``statistic, pvalue = result``), never the full metadata set -- this keeps the
  unpacking contract short and stable.
* ``to_dict()`` returns every populated field; NumPy scalars become Python
  scalars and arrays become lists unless ``array=True`` is passed.
"""
from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, Iterator

import numpy as np

__all__ = ["RichResult", "HypothesisResult", "ArrayResult", "rich_facade"]


def _pyify(value: Any, *, array: bool) -> Any:
    """Make a field value plain-Python friendly for ``to_dict``."""
    if isinstance(value, np.ndarray):
        return value if array else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    """True for None or NaN — the "not populated" sentinel in to_dict/info."""
    return value is None or (isinstance(value, float) and math.isnan(value))


class RichResult:
    """Mixin giving a dataclass the bunker-stats result protocol.

    Subclasses must be dataclasses and set two class attributes:

    ``_fields``
        Tuple of field names, in order, exposed by iteration and indexing.
        These are the primary outputs a user would tuple-unpack.
    ``_title``
        Human-readable title used by :meth:`info`.
    """

    _fields: tuple[str, ...] = ()
    _title: str = "Result"
    # Misuse-prevention notes ("small sample", "approximate p-value", ...).
    # Class-level default; add_warning() creates the per-instance list lazily
    # so dataclass __init__ signatures stay untouched.
    _warnings: tuple = ()

    @property
    def warnings(self) -> list:
        """Misuse-prevention warnings attached to this result (may be empty)."""
        return list(self._warnings)

    def add_warning(self, message: str) -> None:
        """Attach a warning shown by ``info()`` and included in ``to_dict()``."""
        self._warnings = tuple(self._warnings) + (message,)

    # -- tuple-like access ------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        return (getattr(self, name) for name in self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index: int | slice) -> Any:
        values = [getattr(self, name) for name in self._fields]
        return values[index]

    # -- serialization ----------------------------------------------------
    def to_dict(self, *, array: bool = False) -> dict[str, Any]:
        """Return every populated field as a plain dict.

        NumPy scalars are converted to Python scalars. NumPy arrays are
        converted to lists unless ``array=True`` (in which case they are kept
        as ``ndarray`` for further numeric work — not JSON-serializable).
        Fields that are ``None`` are omitted so the dict reflects only what was
        actually computed.
        """
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if value is None:
                continue
            out[f.name] = _pyify(value, array=array)
        if self._warnings:
            out["warnings"] = list(self._warnings)
        return out

    def __repr__(self) -> str:
        primary = ", ".join(
            f"{name}={_fmt_scalar(getattr(self, name))}" for name in self._fields
        )
        return f"{type(self).__name__}({primary})"

    # -- info() scaffolding ----------------------------------------------
    def _info_rows(self) -> list[tuple[str, Any]]:
        """(label, value) rows rendered by :meth:`info`. Override in subclasses."""
        return []

    def _warning_lines(self) -> list[str]:
        if not self._warnings:
            return []
        return [""] + [f"Warning: {w}" for w in self._warnings]

    def info(self) -> str:
        """Pretty, multi-line human summary."""
        lines = [self._title, "=" * 60]
        for label, value in self._info_rows():
            if _is_missing(value):
                continue
            lines.append(f"{label:<20}{_fmt_scalar(value)}")
        lines += self._warning_lines()
        return "\n".join(lines)


class HypothesisResult(RichResult):
    """A :class:`RichResult` for tests exposing ``statistic`` and ``pvalue``.

    Subclasses set ``_h0`` (a short phrase naming the null hypothesis) so the
    generic :meth:`conclusion` reads naturally, and may override
    :meth:`_detail_rows` to add test-specific lines (df, n, method, ...).
    """

    _h0: str = "the null hypothesis"
    _fields = ("statistic", "pvalue")

    # Subclasses are expected to have `.statistic` and `.pvalue` attributes.
    statistic: float
    pvalue: float | None

    def is_significant(self, alpha: float = 0.05) -> bool:
        """True when the p-value is below ``alpha`` (None/NaN -> False)."""
        p = getattr(self, "pvalue", None)
        return not _is_missing(p) and p < alpha

    def conclusion(self, alpha: float = 0.05) -> str:
        """One-line verdict at significance level ``alpha``."""
        p = getattr(self, "pvalue", None)
        if _is_missing(p):
            return f"Inconclusive: no p-value available (compare statistic to critical values)"
        if p < alpha:
            return f"Reject {self._h0} at alpha={alpha:g} (p={p:.4g})"
        return f"Fail to reject {self._h0} at alpha={alpha:g} (p={p:.4g})"

    def _detail_rows(self) -> list[tuple[str, Any]]:
        """Extra (label, value) rows beyond statistic/p-value. Override me."""
        return []

    def info(self) -> str:
        lines = [self._title, "=" * 60, f"{'Statistic:':<20}{self.statistic:.6f}"]
        if not _is_missing(getattr(self, "pvalue", None)):
            lines.append(f"{'P-value:':<20}{self.pvalue:.6f}")
        for label, value in self._detail_rows():
            if _is_missing(value):
                continue
            lines.append(f"{label:<20}{_fmt_scalar(value)}")
        lines += ["", f"Conclusion: {self.conclusion()}"]
        lines += self._warning_lines()
        return "\n".join(lines)


class ArrayResult(RichResult):
    """A :class:`RichResult` whose primary payload is an array.

    For results like a correlation matrix or an outlier mask, tuple-unpacking a
    handful of scalars makes no sense — the natural ergonomics are *array-like*.
    Subclasses set ``_array_field`` (the dataclass field holding the ndarray) and
    this base makes the object behave like that array for the common operations
    while still offering ``to_dict()`` / ``info()`` and any metadata fields.

    Notably ``np.asarray(result)`` returns the payload with no copy, so a result
    object drops straight into NumPy code.
    """

    _array_field: str = ""

    def _payload(self) -> np.ndarray:
        return np.asarray(getattr(self, self._array_field))

    def __array__(self, dtype=None):
        arr = self._payload()
        return arr.astype(dtype) if dtype is not None else arr

    def __iter__(self) -> Iterator[Any]:
        return iter(self._payload())

    def __len__(self) -> int:
        return len(self._payload())

    def __getitem__(self, index):
        return self._payload()[index]

    @property
    def shape(self):
        return self._payload().shape

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, {self._array_field}=...)"


def _fmt_scalar(value: Any) -> str:
    """Compact display for a scalar/array field in repr and info()."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    return str(value)


def rich_facade(raw, builder, *, name=None, doc_note=True, builder_kwargs=()):
    """Wrap a raw (dict/tuple-returning) function with an opt-in ``rich`` flag.

    The returned wrapper forwards **every** positional and keyword argument to
    ``raw`` unchanged, so the default call is byte-for-byte backward compatible.
    Only ``rich`` (and any names in ``builder_kwargs``) are intercepted:

    * ``rich=False`` (default) -> return exactly what ``raw`` returns.
    * ``rich=True``            -> return ``builder(raw_result, args, kwargs)``.

    Parameters
    ----------
    raw : callable
        The underlying function (e.g. a Rust binding) returning a dict/tuple.
    builder : callable
        ``builder(raw_result, args, kwargs) -> RichResult``. Receives the raw
        result plus the original call's positional/keyword args (with any
        ``builder_kwargs`` folded in) so it can add input-derived metadata.
    name : str, optional
        ``__name__`` for the wrapper; defaults to ``raw``'s name.
    builder_kwargs : iterable of str
        Keyword names that are meant for ``builder`` only and must NOT be passed
        to ``raw`` (e.g. ``columns`` for a matrix result). They are popped from
        the call, take effect only with ``rich=True``, and are handed to the
        builder via its ``kwargs`` argument.
    """
    fn_name = name or getattr(raw, "__name__", "wrapped")
    _builder_only = set(builder_kwargs)

    def wrapper(*args, rich: bool = False, **kwargs):
        extra = {k: kwargs.pop(k) for k in list(kwargs) if k in _builder_only}
        result = raw(*args, **kwargs)
        if not rich:
            return result
        return builder(result, args, {**kwargs, **extra})

    wrapper.__name__ = fn_name
    wrapper.__qualname__ = fn_name
    wrapper.__wrapped__ = raw
    base_doc = getattr(raw, "__doc__", None) or ""
    if doc_note:
        note = (
            "\n\nPass ``rich=True`` to receive a rich result object "
            "(tuple-unpackable, with ``.to_dict()``, ``.info()`` and "
            "``.conclusion()``) instead of the default return value."
        )
        wrapper.__doc__ = base_doc + note
    else:
        wrapper.__doc__ = base_doc
    return wrapper
