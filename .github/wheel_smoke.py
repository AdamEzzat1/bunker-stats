# -*- coding: utf-8 -*-
"""Smoke test run against the INSTALLED wheel in CI (not the source tree).

Release gate: this is the check that would have caught the 0.2.x packaging
bug where wheels shipped only the compiled extension and `import bunker_stats`
failed for every installed user. Keep it fast and dependency-light: numpy only
(the notebook import must succeed lazily WITHOUT pandas installed).
"""
import os
import sys

# Guard against accidentally importing the repo checkout instead of the wheel.
sys.path = [p for p in sys.path if not os.path.isdir(os.path.join(p, "bunker_stats", "..", "src"))]

import numpy as np  # noqa: E402

import bunker_stats as bs  # noqa: E402

# The wheel must serve the package, not the checkout.
assert "site-packages" in (bs.__file__ or ""), f"imported from {bs.__file__!r}, not the wheel"

# 1) Every public name resolves — no export may point at a missing symbol.
bad = []
for name in bs.__all__:
    try:
        getattr(bs, name)
    except Exception as exc:  # noqa: BLE001 - report every failure kind
        bad.append((name, repr(exc)))
assert not bad, f"__all__ names failed to resolve from the wheel: {bad}"

# 2) Flagship features work end-to-end from the wheel.
x, y = np.random.default_rng(0).normal(size=(2, 60))

res = bs.t_test_2samp(x, y, rich=True)
stat, pval = res
assert res.conclusion() and res.to_dict()["statistic"] == stat

roll = bs.Rolling(x, window=10).result("mean", "std")
rolled = roll.to_dict()
assert len(rolled["values"]["mean"]) > 0 if "values" in rolled else len(rolled) > 0

from bunker_stats.resampling import BootstrapConfig  # noqa: E402

boot = BootstrapConfig(n_resamples=200, random_state=0, return_draws=True).run(x)
assert boot.draws.shape == (200,)
plain = BootstrapConfig(n_resamples=200, random_state=0).run(x)
assert plain == (boot.estimate, boot.ci_lower, boot.ci_upper), "draws path diverged from tuple path"

# 3) Optional layers stay optional: notebook must import without pandas.
import bunker_stats.notebook  # noqa: E402, F401
import bunker_stats.pandas_helpers  # noqa: E402, F401

print(f"wheel smoke OK: {len(bs.__all__)} names, python {sys.version.split()[0]}")
