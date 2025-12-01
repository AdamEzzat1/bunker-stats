import time
import numpy as np

import bunker_stats as bs   # assumes __init__.py re-exports bunker_stats_rs

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ----------------- helpers ----------------- #

def time_call(fn, *args, repeats=3, **kwargs):
    """Return best time over `repeats` runs."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
            out = result
    return best, out


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b))


# ----------------- basic stats vs NumPy ----------------- #

def benchmark_basic_stats(n=5_000_000):
    print("\n=== Basic stats: mean / std / var (n = {:,}) ===".format(n))
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n).astype(np.float64)

    # mean
    t_np, mean_np_val = time_call(np.mean, x)
    t_bs, mean_bs_val = time_call(bs.mean_np, x)
    print(f"mean:  numpy={t_np:.6f}s, bunker={t_bs:.6f}s, diff={abs(mean_np_val - mean_bs_val):.3e}")

    # std (ddof=1 for fair comparison)
    t_np, std_np_val = time_call(np.std, x, ddof=1)
    t_bs, std_bs_val = time_call(bs.std_np, x)
    print(f"std:   numpy={t_np:.6f}s, bunker={t_bs:.6f}s, diff={abs(std_np_val - std_bs_val):.3e}")

    # var (ddof=1)
    t_np, var_np_val = time_call(np.var, x, ddof=1)
    t_bs, var_bs_val = time_call(bs.var_np, x)
    print(f"var:   numpy={t_np:.6f}s, bunker={t_bs:.6f}s, diff={abs(var_np_val - var_bs_val):.3e}")


# ----------------- quantiles / IQR / MAD vs NumPy ----------------- #

def benchmark_quantiles(n=2_000_000):
    print("\n=== Quantiles / IQR / MAD (n = {:,}) ===".format(n))
    rng = np.random.default_rng(123)
    x = rng.standard_normal(n).astype(np.float64)

    # percentile
    q = 0.25
    t_np, q_np = time_call(np.quantile, x, q)
    t_bs, q_bs = time_call(bs.percentile_np, x, q)
    print(f"percentile q={q}: numpy={t_np:.6f}s, bunker={t_bs:.6f}s, diff={abs(q_np - q_bs):.3e}")

    # IQR
    def iqr_numpy(arr):
        q1 = np.quantile(arr, 0.25)
        q3 = np.quantile(arr, 0.75)
        return q1, q3, q3 - q1

    t_np, iqr_np_val = time_call(iqr_numpy, x)
    t_bs, iqr_bs_val = time_call(bs.iqr_np, x)
    diff_q1 = abs(iqr_np_val[0] - iqr_bs_val[0])
    diff_q3 = abs(iqr_np_val[1] - iqr_bs_val[1])
    diff_iqr = abs(iqr_np_val[2] - iqr_bs_val[2])
    print(
        f"IQR:   numpy={t_np:.6f}s, bunker={t_bs:.6f}s, "
        f"diff_q1={diff_q1:.3e}, diff_q3={diff_q3:.3e}, diff_iqr={diff_iqr:.3e}"
    )

    # MAD
    def mad_numpy(arr):
        med = np.median(arr)
        devs = np.abs(arr - med)
        return np.median(devs)

    t_np, mad_np_val = time_call(mad_numpy, x)
    t_bs, mad_bs_val = time_call(bs.mad_np, x)
    print(f"MAD:   numpy={t_np:.6f}s, bunker={t_bs:.6f}s, diff={abs(mad_np_val - mad_bs_val):.3e}")


# ----------------- rolling stats vs pandas ----------------- #

def benchmark_rolling(n=1_000_000, window=252):
    print("\n=== Rolling mean / std (n = {:,}, window = {}) ===".format(n, window))
    rng = np.random.default_rng(777)
    x = rng.standard_normal(n).astype(np.float64)

    # bunker-stats
    t_mean_bs, mean_bs = time_call(bs.rolling_mean_np, x, window)
    t_std_bs, std_bs = time_call(bs.rolling_std_np, x, window)

    print(f"bunker: rolling_mean={t_mean_bs:.6f}s, rolling_std={t_std_bs:.6f}s")

    if not HAS_PANDAS:
        print("pandas not installed; skipping pandas comparison.")
        return

    s = pd.Series(x)

    # rolling mean (align output length with bunker-stats, which returns n - window + 1)
    def pd_rolling_mean(arr, win):
        return pd.Series(arr).rolling(win).mean().to_numpy()[win - 1 :]

    def pd_rolling_std(arr, win):
        # ddof=1 to match sample std
        return pd.Series(arr).rolling(win).std(ddof=1).to_numpy()[win - 1 :]

    t_mean_pd, mean_pd = time_call(pd_rolling_mean, x, window)
    t_std_pd, std_pd = time_call(pd_rolling_std, x, window)

    mean_diff = max_abs_diff(mean_pd, mean_bs)
    std_diff = max_abs_diff(std_pd, std_bs)

    print(f"pandas: rolling_mean={t_mean_pd:.6f}s, rolling_std={t_std_pd:.6f}s")
    print(f"diffs:  mean max_abs_diff={mean_diff:.3e}, std max_abs_diff={std_diff:.3e}")


# ----------------- covariance / correlation vs NumPy ----------------- #

def benchmark_cov_corr_vector(n=2_000_000):
    print("\n=== cov / corr (vector) (n = {:,}) ===".format(n))
    rng = np.random.default_rng(999)
    x = rng.standard_normal(n).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)

    # cov
    t_np_cov, cov_np_val = time_call(np.cov, x, y, ddof=1)
    # np.cov returns 2x2; [0,1] or [1,0] is the cross-cov
    cov_np_scalar = cov_np_val[0, 1]

    t_bs_cov, cov_bs_val = time_call(bs.cov_np, x, y)
    print(f"cov:   numpy={t_np_cov:.6f}s, bunker={t_bs_cov:.6f}s, diff={abs(cov_np_scalar - cov_bs_val):.3e}")

    # corr
    t_np_corr, corr_np_val = time_call(np.corrcoef, x, y)
    corr_np_scalar = corr_np_val[0, 1]

    t_bs_corr, corr_bs_val = time_call(bs.corr_np, x, y)
    print(f"corr:  numpy={t_np_corr:.6f}s, bunker={t_bs_corr:.6f}s, diff={abs(corr_np_scalar - corr_bs_val):.3e}")


def benchmark_cov_corr_matrix(n_samples=200_000, n_features=8):
    print(
        "\n=== cov / corr (matrix) (samples = {:,}, features = {}) ===".format(
            n_samples, n_features
        )
    )
    rng = np.random.default_rng(2024)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    # cov
    t_np_cov, cov_np_val = time_call(
        np.cov, X, rowvar=False, ddof=1
    )  # shape (n_features, n_features)
    t_bs_cov, cov_bs_val = time_call(bs.cov_matrix_np, X)
    cov_diff = max_abs_diff(cov_np_val, cov_bs_val)
    print(f"cov matrix: numpy={t_np_cov:.6f}s, bunker={t_bs_cov:.6f}s, max_abs_diff={cov_diff:.3e}")

    # corr
    t_np_corr, corr_np_val = time_call(
        np.corrcoef, X, rowvar=False
    )  # shape (n_features, n_features)
    t_bs_corr, corr_bs_val = time_call(bs.corr_matrix_np, X)
    corr_diff = max_abs_diff(corr_np_val, corr_bs_val)
    print(f"corr matrix: numpy={t_np_corr:.6f}s, bunker={t_bs_corr:.6f}s, max_abs_diff={corr_diff:.3e}")


# ----------------- more complex bunker-stats functions ----------------- #

def benchmark_complex(n=1_000_000, n_bins=10):
    print("\n=== Complex functions (n = {:,}) ===".format(n))
    rng = np.random.default_rng(314)
    x = rng.standard_normal(n).astype(np.float64)

    # IQR outliers vs a pure-NumPy version
    def iqr_outliers_numpy(arr, k):
        q1 = np.quantile(arr, 0.25)
        q3 = np.quantile(arr, 0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        return (arr < low) | (arr > high)

    k = 1.5
    t_np_iqr, mask_np = time_call(iqr_outliers_numpy, x, k)
    t_bs_iqr, mask_bs = time_call(bs.iqr_outliers_np, x, k)
    mask_diff = np.sum(mask_np != mask_bs)
    print(
        f"iqr_outliers: numpy={t_np_iqr:.6f}s, bunker={t_bs_iqr:.6f}s, "
        f"num_disagreements={mask_diff}"
    )

    # winsorize vs NumPy implementation
    def winsorize_numpy(arr, lower_q, upper_q):
        low = np.quantile(arr, lower_q)
        high = np.quantile(arr, upper_q)
        return np.clip(arr, low, high)

    lower_q, upper_q = 0.05, 0.95
    t_np_win, w_np = time_call(winsorize_numpy, x, lower_q, upper_q)
    t_bs_win, w_bs = time_call(bs.winsorize_np, x, lower_q, upper_q)
    win_diff = max_abs_diff(w_np, w_bs)
    print(
        f"winsorize:  numpy={t_np_win:.6f}s, bunker={t_bs_win:.6f}s, max_abs_diff={win_diff:.3e}"
    )

    # quantile_bins vs pandas.qcut labels (if pandas available)
    if HAS_PANDAS:
        def qcut_bins(arr, n_bins_):
            s = pd.Series(arr)
            # labels = False gives bin indices [0..n_bins_-1] where possible
            return pd.qcut(s, q=n_bins_, labels=False, duplicates="drop").to_numpy()

        t_pd_bins, bins_pd = time_call(qcut_bins, x, n_bins)
        t_bs_bins, bins_bs = time_call(bs.quantile_bins_np, x, n_bins)

        # Align length (qcut can drop NaNs; we don't have NaNs here though)
        bins_pd = bins_pd.astype(np.int32)
        bins_bs = np.asarray(bins_bs, dtype=np.int32)

        # We don't expect exact same bin indices due to subtle quantile differences,
        # but we can report disagreement rate.
        len_min = min(len(bins_pd), len(bins_bs))
        disagreements = np.sum(bins_pd[:len_min] != bins_bs[:len_min])
        print(
            f"quantile_bins vs qcut: pandas={t_pd_bins:.6f}s, bunker={t_bs_bins:.6f}s, "
            f"disagreements={disagreements}/{len_min}"
        )
    else:
        print("pandas not installed; skipping quantile_bins vs qcut comparison.")


# ----------------- main ----------------- #

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    benchmark_basic_stats()
    benchmark_quantiles()
    benchmark_rolling()
    benchmark_cov_corr_vector()
    benchmark_cov_corr_matrix()
    benchmark_complex()
