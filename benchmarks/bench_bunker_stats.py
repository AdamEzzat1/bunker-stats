"""
Benchmarks for bunker-stats vs NumPy/pandas.

Run from project root (with venv/conda env active and bunker-stats installed via `maturin develop`):

    python benchmarks/bench_bunker_stats.py
"""

import time
from typing import Callable, Any, Dict

import numpy as np
import pandas as pd

import bunker_stats as bs


def benchmark(
    name: str,
    fn: Callable[..., Any],
    *args,
    repeats: int = 5,
    warmup: int = 1,
) -> float:
    """
    Run `fn(*args)` multiple times and return best time (seconds).
    """
    # warmup
    for _ in range(warmup):
        fn(*args)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        best = min(best, end - start)
    print(f"{name:30s}: {best*1000:8.2f} ms")
    return best


def bench_1d_stats(n: int = 1_000_000) -> Dict[str, float]:
    print(f"\n=== 1D stats on array of length {n} ===")
    x = np.random.randn(n).astype("float64")

    results: Dict[str, float] = {}

    # mean
    results["numpy_mean"] = benchmark("numpy.mean", np.mean, x)
    results["bunker_mean"] = benchmark("bunker.mean_np", bs.mean_np, x)

    # std (sample, ddof=1) - use keyword arg for ddof
    results["numpy_std"] = benchmark(
        "numpy.std (ddof=1)",
        lambda arr: np.std(arr, ddof=1),
        x,
    )
    results["bunker_std"] = benchmark("bunker.std_np", bs.std_np, x)

    # z-score
    def numpy_zscore(arr: np.ndarray) -> np.ndarray:
        m = arr.mean()
        s = arr.std(ddof=1)
        return (arr - m) / s

    results["numpy_zscore"] = benchmark("numpy z-score", numpy_zscore, x)
    results["bunker_zscore"] = benchmark("bunker.zscore_np", bs.zscore_np, x)

    return results


def bench_rolling(n: int = 1_000_000, window: int = 50) -> Dict[str, float]:
    print(f"\n=== Rolling mean on length {n}, window={window} ===")
    x = np.random.randn(n).astype("float64")
    # s = pd.Series(x)  # we reconstruct inside the lambda below

    results: Dict[str, float] = {}

    results["pandas_rolling_mean"] = benchmark(
        "pandas.Series.rolling().mean",
        lambda arr: pd.Series(arr).rolling(window).mean().to_numpy(),
        x,
    )

    results["bunker_rolling_mean"] = benchmark(
        "bunker.rolling_mean_np",
        bs.rolling_mean_np,
        x,
        window,
    )

    return results


def bench_corr_matrix(n_rows: int = 100_000, n_cols: int = 10) -> Dict[str, float]:
    print(f"\n=== Corr matrix on df shape=({n_rows}, {n_cols}) ===")
    data = np.random.randn(n_rows, n_cols).astype("float64")
    cols = [f"col_{i}" for i in range(n_cols)]
    df = pd.DataFrame(data, columns=cols)

    results: Dict[str, float] = {}

    # pandas corr
    results["pandas_corr"] = benchmark(
        "pandas.DataFrame.corr",
        lambda d: d.corr().to_numpy(),
        df,
    )

    # bunker corr_matrix_np (works on numpy array)
    results["bunker_corr"] = benchmark(
        "bunker.corr_matrix_np",
        bs.corr_matrix_np,
        data,
    )

    return results


def sanity_checks():
    """
    Run small sanity checks to ensure NumPy/pandas and bunker-stats agree numerically.
    """
    print("\n=== Sanity checks (small arrays/DataFrames) ===")

    x = np.array([1.0, 2.0, 3.0, 10.0], dtype="float64")

    m_np = np.mean(x)
    m_bs = bs.mean_np(x)
    print("mean numpy vs bunker:", m_np, m_bs)

    std_np = np.std(x, ddof=1)
    std_bs = bs.std_np(x)
    print("std numpy(ddof=1) vs bunker:", std_np, std_bs)

    z_np = (x - m_np) / std_np
    z_bs = bs.zscore_np(x)
    print("z-score numpy vs bunker:", z_np, z_bs)

    # corr matrix sanity
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 3.0, 4.0, 5.0],
            "c": [10.0, -1.0, 0.0, 3.0],
        }
    )
    corr_pd = df.corr().to_numpy()
    corr_bs = bs.corr_matrix_np(df.to_numpy(dtype="float64"))

    print("pandas corr:\n", corr_pd)
    print("bunker corr:\n", corr_bs)


def main():
    sanity_checks()

    # You can tweak sizes here if you want heavier/lighter runs
    bench_1d_stats(n=1_000_000)
    bench_rolling(n=1_000_000, window=50)
    bench_corr_matrix(n_rows=100_000, n_cols=10)


if __name__ == "__main__":
    main()
