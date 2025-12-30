import numpy as np

def show(out: dict, ref) -> str:
    return (
        f"out={out}\n"
        f"ref_stat={ref.statistic}, ref_p={ref.pvalue}"
    )

def isclose(a, b, rtol, atol):
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))
