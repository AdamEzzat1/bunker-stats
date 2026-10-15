# bunker_stats/resampling/__init__.py
"""
Resampling methods with ergonomic config objects.

This module provides two interfaces:

1. **Config objects** (recommended for complex workflows):
   - BootstrapConfig, BootstrapCorrConfig
   - StratifiedBootstrapConfig  # v0.3: NEW
   - ClusteredBootstrapConfig   # v0.3: NEW
   - WildBootstrapOLSConfig     # v0.3: NEW
   - PermutationConfig
   - JackknifeConfig
   
2. **Convenience functions** (for quick one-liners):
   - bootstrap(), bootstrap_corr()
   - bootstrap_stratified()  # v0.3: NEW
   - bootstrap_clustered()   # v0.3: NEW
   - wild_bootstrap_ols()    # v0.3: NEW
   - permutation_test()
   - jackknife()

Both interfaces call the same fast Rust kernels. Config objects add:
- Input validation with helpful error messages
- Consistent defaults
- Optional NaN handling (pre-filter in Python)

Examples
--------
Config object approach:
>>> from bunker_stats.resampling import BootstrapConfig
>>> config = BootstrapConfig(n_resamples=5000, conf=0.99, random_state=42)
>>> estimate, lower, upper = config(data)

Stratified bootstrap (v0.3):
>>> from bunker_stats.resampling import StratifiedBootstrapConfig
>>> config = StratifiedBootstrapConfig(n_resamples=5000, random_state=42)
>>> strata = np.array([0, 0, 1, 1, 1, 2, 2])  # Stratum labels
>>> estimate, lower, upper = config.run(data, strata)

Functional approach:
>>> from bunker_stats.resampling import bootstrap
>>> estimate, lower, upper = bootstrap(data, n_resamples=5000, conf=0.99, random_state=42)

For backward compatibility, the original flat functions remain available:
>>> import bunker_stats as bsr
>>> bsr.bootstrap_ci(data, stat="mean", n_resamples=1000, conf=0.95)
"""

from .config import (
    # Config dataclasses
    BootstrapConfig,
    BootstrapCorrConfig,
    StratifiedBootstrapConfig,  # v0.3: NEW
    ClusteredBootstrapConfig,   # v0.3: NEW
    WildBootstrapOLSConfig,     # v0.3: NEW
    BcaCorrConfig,              # v0.3: NEW
    BcaMeanDiffConfig,          # v0.3: NEW
    BayesianBootstrapSEConfig,  # v0.3: NEW
    PermutationConfig,
    JackknifeConfig,
    # v0.3: Extended permutation tests
    PermutationTestAnovaConfig,
    PermutationTestMannWhitneyConfig,
    PermutationTestKruskalWallisConfig,
    PermutationTestPartialCorrConfig,
    # v0.3: Multiple testing corrections
    BonferroniConfig,
    HolmConfig,
    FDRConfig,

    # Rich result objects (shared protocol: unpacking, to_dict, info, conclusion)
    BootstrapResult,
    PermutationTestResult,

    # Convenience functions
    bootstrap,
    bootstrap_corr,
    bootstrap_stratified,  # v0.3: NEW
    bootstrap_clustered,   # v0.3: NEW
    wild_bootstrap_ols,    # v0.3: NEW
    bca_corr,              # v0.3: NEW
    bca_mean_diff,         # v0.3: NEW
    bayesian_bootstrap_se, # v0.3: NEW
    permutation_test,
    jackknife,
    # v0.3: Extended permutation convenience functions
    permutation_test_anova,
    permutation_test_mann_whitney,
    permutation_test_kruskal_wallis,
    permutation_test_partial_corr,
    # v0.3: Multiple testing convenience function
    multipletests,
)

__all__ = [
    # Config objects
    "BootstrapConfig",
    "BootstrapCorrConfig",
    "StratifiedBootstrapConfig",  # v0.3: NEW
    "ClusteredBootstrapConfig",   # v0.3: NEW
    "WildBootstrapOLSConfig",     # v0.3: NEW
    "BcaCorrConfig",              # v0.3: NEW
    "BcaMeanDiffConfig",          # v0.3: NEW
    "BayesianBootstrapSEConfig",  # v0.3: NEW
    "PermutationConfig",
    "JackknifeConfig",
    # v0.3: Extended permutation tests
    "PermutationTestAnovaConfig",
    "PermutationTestMannWhitneyConfig",
    "PermutationTestKruskalWallisConfig",
    "PermutationTestPartialCorrConfig",
    # v0.3: Multiple testing corrections
    "BonferroniConfig",
    "HolmConfig",
    "FDRConfig",

    # Rich result objects
    "BootstrapResult",
    "PermutationTestResult",

    # Convenience functions
    "bootstrap",
    "bootstrap_corr",
    "bootstrap_stratified",  # v0.3: NEW
    "bootstrap_clustered",   # v0.3: NEW
    "wild_bootstrap_ols",    # v0.3: NEW
    "bca_corr",              # v0.3: NEW
    "bca_mean_diff",         # v0.3: NEW
    "bayesian_bootstrap_se", # v0.3: NEW
    "permutation_test",
    "jackknife",
    # v0.3: Extended permutation convenience functions
    "permutation_test_anova",
    "permutation_test_mann_whitney",
    "permutation_test_kruskal_wallis",
    "permutation_test_partial_corr",
    # v0.3: Multiple testing convenience function
    "multipletests",
]