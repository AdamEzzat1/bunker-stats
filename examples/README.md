# Examples

## Notebook UX gallery

[`notebook_gallery.html`](./notebook_gallery.html) is a self-contained visual
gallery of every helper in `bunker_stats.notebook`, run on one small "sensor
readings" table (with a fat outlier, missing values, an ∞, a correlated pair
and a constant column) and rendered exactly as it appears in a notebook.

Open the file in any browser. It shows, for each helper:

- **Reports** (`→ DataFrame`) — `robust_summary`, `describe_fast`,
  `outlier_report`, `normality_report`, `correlation_report`,
  `missingness_report`, `rolling_report`, `bootstrap_ci_report`
- **Transforms** (`→ DataFrame (+cols)`) — `scale_columns`, `winsorize_columns`
- **Stylers** (`→ Styler`) — `outlier_style`, `corr_heatmap`,
  `style_significance`, `style_effect_size`, `demean_style`, `zscore_style`,
  `iqr_outlier_style`

The rendered HTML is real output — the colored cells come straight from the
Rust kernels' decisions, so the gallery doubles as a correctness snapshot.

### Regenerating

```bash
pip install "bunker-stats-rs[notebook]"
python examples/generate_gallery.py
```

This rewrites `notebook_gallery.html`. Edit
[`generate_gallery.py`](./generate_gallery.py) to change the sample data or add
cards.
