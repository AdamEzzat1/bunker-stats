import numpy as np
import pandas as pd
import time
import bunker_stats as bs

def bench(name, fn, *args, ref_fn=None, ref_args=()):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000
    print(f"{name}: {dt:.3f} ms")
    if ref_fn is not None:
        ref_out = ref_fn(*ref_args)
        ok = np.allclose(out, ref_out, equal_nan=True)
        max_diff = np.nanmax(np.abs(out - ref_out))
        print(f"  allclose={ok}, max_abs_diff={max_diff}")
    return out

def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100_000).astype("float64")
    # sprinkle NaNs
    mask = rng.random(size=x.size) < 0.1
    x[mask] = np.nan

    print("=== scalar nan stats ===")
    bench(
        "mean_nan_np vs np.nanmean",
        bs.mean_nan_np,
        x,
        ref_fn=np.nanmean,
        ref_args=(x,),
    )
    bench(
        "std_nan_np vs np.nanstd(ddof=1)",
        bs.std_nan_np,
        x,
        ref_fn=lambda v: np.nanstd(v, ddof=1),
        ref_args=(x,),
    )

    print("\n=== rolling nan stats ===")
    s = pd.Series(x)
    window = 50

    # bunker rolling mean
    bs_rm = bench(
        "rolling_mean_nan_np",
        bs.rolling_mean_nan_np,
        x,
        window,
        ref_fn=lambda v: s.rolling(window, min_periods=1).mean().to_numpy(),
        ref_args=(x,),
    )

    # bunker rolling zscore
    def pandas_rolling_zscore(arr, window):
        s = pd.Series(arr)
        roll = s.rolling(window, min_periods=1)
        m = roll.mean()
        sd = roll.std(ddof=1)
        return ((s - m) / sd).to_numpy()

    bs_rz = bench(
        "rolling_zscore_nan_np",
        bs.rolling_zscore_nan_np,
        x,
        window,
        ref_fn=pandas_rolling_zscore,
        ref_args=(x, window),
    )

if __name__ == "__main__":
    main()
