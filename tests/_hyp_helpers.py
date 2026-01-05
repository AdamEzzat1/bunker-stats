# tests/_hyp_helpers.py
import numpy as np
from hypothesis import strategies as st

finite_f64 = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

arrays_f64 = st.lists(
    finite_f64,
    min_size=5,
    max_size=300,
).map(lambda xs: np.asarray(xs, dtype=np.float64))

arrays_f64_with_nans = st.lists(
    st.one_of(finite_f64, st.just(np.nan)),
    min_size=5,
    max_size=300,
).map(lambda xs: np.asarray(xs, dtype=np.float64))
