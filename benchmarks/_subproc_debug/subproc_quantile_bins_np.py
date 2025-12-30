
import json, os, time
import numpy as np

import faulthandler
faulthandler.enable()

import bunker_stats as bs

CASE = {"fn_name": "quantile_bins_np", "n": 200000, "p": 32, "seed": 0, "kind": "vec_f64", "repeats": 3, "warmup": 1}

def _make_inputs(case):
    n = int(case.get("n", 200000))
    p = int(case.get("p", 32))
    seed = int(case.get("seed", 0))
    rng = np.random.default_rng(seed)
    kind = case.get("kind", "vec_f64")

    def vec_f64():
        return rng.normal(size=n).astype(np.float64)

    def vec_nan_f64():
        x = rng.normal(size=n).astype(np.float64)
        x[rng.random(size=n) < 0.05] = np.nan
        return x

    def mat_f64():
        return rng.normal(size=(n, p)).astype(np.float64)

    def pair_vec_f64():
        x = rng.normal(size=n).astype(np.float64)
        y = rng.normal(loc=0.2, size=n).astype(np.float64)
        return x, y

    def pair_vec_nan_f64():
        x, y = pair_vec_f64()
        x[rng.random(size=n) < 0.05] = np.nan
        y[rng.random(size=n) < 0.05] = np.nan
        return x, y

    if kind == "vec_f64":
        return {"x": vec_f64()}
    if kind == "vec_nan_f64":
        return {"x": vec_nan_f64()}
    if kind == "mat_f64":
        return {"X": mat_f64()}
    if kind == "pair_vec_f64":
        x, y = pair_vec_f64()
        return {"x": x, "y": y}
    if kind == "pair_vec_nan_f64":
        x, y = pair_vec_nan_f64()
        return {"x": x, "y": y}
    if kind == "chi2_gof":
        obs = np.array([10, 12, 9, 11, 8], dtype=np.float64)
        exp = np.array([10, 10, 10, 10, 10], dtype=np.float64)
        return {"obs": obs, "exp": exp}
    if kind == "chi2_ind":
        tab = np.array([[10, 20, 30], [6, 9, 17]], dtype=np.float64)
        return {"tab": tab}
    return {"x": vec_f64()}

def _call_bs(case, inputs):
    fn = getattr(bs, case["fn_name"], None)
    if fn is None:
        return {"status":"skip_missing"}

    name = case["fn_name"]
    k = case.get("kind", "vec_f64")

    # IMPORTANT: call with positional args to avoid keyword-arg incompatibilities.
    if k in ("vec_f64","vec_nan_f64"):
        x = inputs["x"]
        if name == "percentile_np":
            return {"status":"ok","value": fn(x, float(case.get("q",95.0)))}
        if name in ("diff_np","pct_change_np"):
            return {"status":"ok","value": fn(x, int(case.get("periods", 1)))}
        if name in ("rolling_mean_np","rolling_std_np","rolling_var_np","rolling_mean_std_np","rolling_zscore_np",
                    "rolling_mean_nan_np","rolling_std_nan_np","rolling_zscore_nan_np"):
            return {"status":"ok","value": fn(x, int(case.get("window", 64)))}
        if name == "ewma_np":
            return {"status":"ok","value": fn(x, float(case.get("alpha", 0.2)))}
        if name == "t_test_1samp_np":
            return {"status":"ok","value": fn(x, float(case.get("mu",0.0)), str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x)}

    if k == "mat_f64":
        X = inputs["X"]
        if name in ("rolling_mean_axis0_np","rolling_std_axis0_np","rolling_mean_std_axis0_np"):
            return {"status":"ok","value": fn(X, int(case.get("window", 64)))}
        return {"status":"ok","value": fn(X)}

    if k in ("pair_vec_f64","pair_vec_nan_f64"):
        x = inputs["x"]; y = inputs["y"]
        if name in ("rolling_cov_np","rolling_corr_np","rolling_cov_nan_np","rolling_corr_nan_np"):
            return {"status":"ok","value": fn(x, y, int(case.get("window", 64)))}
        if name == "t_test_2samp_np":
            return {"status":"ok","value": fn(x, y, bool(case.get("equal_var", False)), str(case.get("alternative","two-sided")))}
        return {"status":"ok","value": fn(x, y)}

    if k == "chi2_gof":
        return {"status":"ok","value": fn(inputs["obs"], inputs["exp"])}
    if k == "chi2_ind":
        return {"status":"ok","value": fn(inputs["tab"])}

    return {"status":"skip_unknown"}

def main():
    case = dict(CASE)
    inputs = _make_inputs(case)
    warmup = int(case.get("warmup", 1))
    repeats = int(case.get("repeats", 3))

    try:
        for _ in range(warmup):
            r = _call_bs(case, inputs)
            if r.get("status") != "ok":
                print(json.dumps(r))
                return

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = _call_bs(case, inputs)
            t1 = time.perf_counter()
            if r.get("status") != "ok":
                print(json.dumps(r))
                return
            times.append(t1 - t0)

        out = {
            "status": "ok",
            "fn": case["fn_name"],
            "median_s": float(np.median(times)) if times else float("nan"),
            "mean_s": float(np.mean(times)) if times else float("nan"),
            "cv": float(np.std(times, ddof=1) / np.mean(times)) if len(times) > 1 and np.mean(times) else float("nan"),
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"status":"bs_error","error":repr(e)}))

if __name__ == "__main__":
    main()
