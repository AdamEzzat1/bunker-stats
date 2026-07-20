"""Generate a visual gallery of every bunker_stats.notebook helper.

Runs each helper on realistic sample data and embeds its *real* rendered
output (Styler HTML for stylers, styled DataFrame HTML for reports) into a
single self-contained page: examples/notebook_gallery.html

Run with the notebook extra installed:

    pip install "bunker-stats-rs[notebook]"
    python examples/generate_gallery.py
"""
from __future__ import annotations

import html as _html
import pathlib

import numpy as np
import pandas as pd

from bunker_stats import notebook as nb

OUT = pathlib.Path(__file__).with_name("notebook_gallery.html")

# ----------------------------------------------------------------------
# Sample data: a small "sensor readings" table with realistic warts --
# a heavy-tailed column, a clean normal column, a correlated pair, a
# constant column, missing values and an infinity.
# ----------------------------------------------------------------------
rng = np.random.default_rng(20260720)
N = 40

price = rng.normal(100, 8, N)
price[5] = 210.0          # a fat outlier
price[12] = 38.0          # a low outlier
price[20] = np.nan        # missing

latency = rng.normal(50, 5, N)          # clean, ~normal
demand = 3.0 * price + rng.normal(0, 12, N)   # correlated with price
demand[8] = np.inf                       # a bad sensor reading
temperature = np.full(N, 21.5)           # constant (zero variance)
region = np.where(np.arange(N) % 2 == 0, "north", "south")

DF = pd.DataFrame(
    {
        "price": price,
        "latency": latency,
        "demand": demand,
        "temperature": temperature,
        "region": region,
    }
)

# A separate "results table" for the significance / effect-size stylers.
RESULTS = pd.DataFrame(
    {
        "comparison": ["A vs B", "A vs C", "B vs C", "C vs D", "D vs E"],
        "pvalue": [0.0003, 0.011, 0.048, 0.21, 0.83],
        "cohens_d": [1.35, -0.72, 0.34, 0.11, 0.02],
    }
)


def styler_html(styler) -> str:
    """Rendered HTML for a Styler, with a stable uuid so ids don't collide."""
    return styler.set_uuid("g" + str(abs(hash(styler)) % 10_000) + "_").to_html()


def frame_html(df, *, fmt="{:.3f}") -> str:
    """A plain report DataFrame rendered as a light Styler for consistency."""
    numeric = df.select_dtypes(include="number").columns
    styler = (
        df.style
        .format({c: fmt for c in numeric}, na_rep="·")
        .set_uuid("g" + str(abs(hash(df.to_numpy().tobytes())) % 10_000) + "_")
    )
    return styler.to_html()


# ----------------------------------------------------------------------
# Each entry: (anchor, title, kind, one-line-call, description, html)
# kind drives the little colored tag: report / style / columns
# ----------------------------------------------------------------------
CARDS = []


def add(title, kind, call, desc, rendered):
    anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
    CARDS.append((anchor, title, kind, call, desc, rendered))


# --- Reports ----------------------------------------------------------
add(
    "robust_summary", "report",
    "nb.robust_summary(df)",
    "Robust + classical descriptives per column: count, mean, std, median, "
    "MAD, IQR, Qn scale, trimmed mean, skew, kurtosis. Non-finite values are "
    "dropped and counted in n_missing.",
    frame_html(nb.robust_summary(DF)),
)
add(
    "describe_fast", "report",
    "nb.describe_fast(df)",
    "A faster, richer df.describe().T backed by the Rust kernels. Adds a robust "
    "block (MAD, IQR, Qn, trimmed mean, skew, kurtosis) alongside the quartiles.",
    frame_html(nb.describe_fast(DF)),
)
add(
    "outlier_report", "report",
    'nb.outlier_report(df, method="iqr", k=1.5)',
    "Per-column outlier counts, percentages and fence bounds. Methods: iqr, "
    "zscore, robust_zscore (median/MAD fences).",
    frame_html(nb.outlier_report(DF, method="iqr")),
)
add(
    "normality_report", "report",
    "nb.normality_report(df)",
    "Jarque-Bera and Anderson-Darling diagnostics. The normal/conclusion "
    "verdict uses the JB p-value; A-D reports its statistic only (no p-value "
    "in the kernel).",
    frame_html(nb.normality_report(DF), fmt="{:.4f}"),
)
add(
    "correlation_report", "report",
    'nb.correlation_report(df, pvalues=True)',
    "Correlation between numeric columns using pairwise-complete rows. Long "
    "form (pvalues=True) gives one row per pair with the test statistic and "
    "p-value; the default returns a square matrix.",
    frame_html(nb.correlation_report(DF, pvalues=True), fmt="{:.4f}"),
)
add(
    "missingness_report", "report",
    "nb.missingness_report(df)",
    "The data-quality audit: separates NaN from ±inf from finite, per column, "
    "for every dtype. The one helper that counts rather than drops.",
    frame_html(nb.missingness_report(DF), fmt="{:.1f}"),
)
add(
    "rolling_report", "report",
    'nb.rolling_report(df, "price", window=5)',
    "Rolling-window features via the fused Rust kernel (all stats in one pass). "
    "Right-aligned and index-preserving. Showing the first 10 rows.",
    frame_html(nb.rolling_report(DF, "price", 5).head(10)),
)
add(
    "bootstrap_ci_report", "report",
    'nb.bootstrap_ci_report(df, stat="mean", random_state=0)',
    "Bootstrap point estimate and confidence interval per column, via "
    "BootstrapConfig. Deterministic given random_state.",
    frame_html(nb.bootstrap_ci_report(DF, n_resamples=2000, random_state=0)),
)

# --- Transforms -------------------------------------------------------
add(
    "scale_columns", "columns",
    'nb.scale_columns(df, ["price", "latency"], method="robust")',
    "Batch scaling (robust / zscore / minmax). Fit on finite values only, "
    "scattered back so NaN positions and row order are preserved. Showing the "
    "new columns beside their sources.",
    frame_html(
        nb.scale_columns(DF, ["price", "latency"], method="robust")
        [["price", "price_robust", "latency", "latency_robust"]].head(10)
    ),
)
add(
    "winsorize_columns", "columns",
    'nb.winsorize_columns(df, ["price"], lower_q=0.05, upper_q=0.95)',
    "Batch winsorization — clip tails at the given quantile fractions. Note "
    "how the 210 and 38 outliers are pulled to the fences while other rows are "
    "untouched.",
    frame_html(
        nb.winsorize_columns(DF, ["price"], lower_q=0.05, upper_q=0.95)
        [["price", "price_winsor"]].head(14)
    ),
)

# --- Stylers ----------------------------------------------------------
add(
    "outlier_style", "style",
    'nb.outlier_style(df, ["price", "demand"])',
    "Highlights outlier cells across many numeric columns at once (red). "
    "Non-finite cells are left unstyled. Showing the first 16 rows.",
    styler_html(nb.outlier_style(DF.head(16), ["price", "demand"])),
)
add(
    "corr_heatmap", "style",
    "nb.corr_heatmap(df)",
    "Correlation matrix as a diverging background-gradient Styler "
    "(needs matplotlib). price↔demand shows the planted correlation.",
    styler_html(nb.corr_heatmap(DF)),
)
add(
    "style_significance", "style",
    'nb.style_significance(results, alpha=0.05)',
    "Shades a results table by significance tier (p<α/50, p<α/5, p<α, then "
    "non-significant). NaN p-values stay unstyled.",
    styler_html(nb.style_significance(RESULTS)),
)
add(
    "style_effect_size", "style",
    'nb.style_effect_size(results, "cohens_d")',
    "Shades an effect-size column by |magnitude| against thresholds "
    "(default Cohen's 0.2/0.5/0.8: negligible→small→medium→large).",
    styler_html(nb.style_effect_size(RESULTS, "cohens_d")),
)
add(
    "demean_style", "style",
    'nb.demean_style(df, "price")',
    "Legacy single-column styler (hardened): adds a demeaned column and colors "
    "each cell above (green) / below (red) the mean. Showing 14 rows.",
    styler_html(nb.demean_style(DF.head(14), "price")),
)
add(
    "zscore_style", "style",
    'nb.zscore_style(df, "price", threshold=2.0)',
    "Adds a z-score column and highlights scores beyond ±threshold "
    "(high = orange, low = blue). Showing 14 rows.",
    styler_html(nb.zscore_style(DF.head(14), "price", threshold=2.0)),
)
add(
    "iqr_outlier_style", "style",
    'nb.iqr_outlier_style(df, "price", k=1.5)',
    "Single-column IQR outlier highlight — a thin wrapper over the "
    "multi-column outlier_style. Showing 16 rows.",
    styler_html(nb.iqr_outlier_style(DF.head(16), "price", k=1.5)),
)


# ----------------------------------------------------------------------
# Assemble the page
# ----------------------------------------------------------------------
KIND_LABEL = {
    "report": ("→ DataFrame", "#3949ab"),
    "style": ("→ Styler", "#00897b"),
    "columns": ("→ DataFrame (+cols)", "#8e24aa"),
}

nav_groups = {"report": [], "columns": [], "style": []}
for anchor, title, kind, *_ in CARDS:
    nav_groups[kind].append((anchor, title))

def nav_section(kind, label):
    items = "".join(
        f'<a href="#{a}"><code>{t}</code></a>' for a, t in nav_groups[kind]
    )
    return f'<div class="nav-group"><h4>{label}</h4>{items}</div>'

cards_html = []
for anchor, title, kind, call, desc, rendered in CARDS:
    tag_text, tag_color = KIND_LABEL[kind]
    cards_html.append(f"""
    <section class="card" id="{anchor}">
      <div class="card-head">
        <h2><code>{title}</code></h2>
        <span class="tag" style="background:{tag_color}">{tag_text}</span>
      </div>
      <pre class="call"><span class="prompt">&gt;&gt;&gt;</span> {_html.escape(call)}</pre>
      <p class="desc">{_html.escape(desc)}</p>
      <div class="render">{rendered}</div>
    </section>""")

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bunker-stats · Notebook UX gallery</title>
<style>
  :root {{
    --bg:#f6f7fb; --card:#ffffff; --ink:#1a1c24; --muted:#5b6072;
    --line:#e5e7f0; --code:#2d2f39; --accent:#3949ab;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  }}
  header.hero {{
    padding:40px 24px 28px; background:linear-gradient(120deg,#2b2f52,#3949ab);
    color:#fff;
  }}
  header.hero .wrap {{ max-width:1080px; margin:0 auto; }}
  header.hero h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.3px; }}
  header.hero p {{ margin:0; opacity:.9; max-width:70ch; }}
  header.hero .install {{
    margin-top:16px; display:inline-block; background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25); border-radius:7px;
    padding:7px 12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:13px;
  }}
  .layout {{ max-width:1080px; margin:0 auto; padding:24px; display:grid;
             grid-template-columns:220px 1fr; gap:28px; align-items:start; }}
  nav {{ position:sticky; top:18px; font-size:13px; }}
  .nav-group {{ margin-bottom:16px; }}
  .nav-group h4 {{ margin:0 0 6px; text-transform:uppercase; letter-spacing:.06em;
                   font-size:11px; color:var(--muted); }}
  nav a {{ display:block; padding:3px 8px; border-radius:5px; color:var(--code);
           text-decoration:none; }}
  nav a:hover {{ background:#eceefb; color:var(--accent); }}
  nav code {{ font-size:12.5px; }}
  main {{ min-width:0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
           padding:20px 22px; margin-bottom:22px; box-shadow:0 1px 2px rgba(20,22,40,.04);
           scroll-margin-top:18px; }}
  .card-head {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
  .card-head h2 {{ margin:0; font-size:18px; }}
  .card-head code {{ color:var(--accent); }}
  .tag {{ color:#fff; font-size:11px; font-weight:600; padding:3px 9px;
          border-radius:20px; letter-spacing:.02em; }}
  pre.call {{ background:#1f2130; color:#e7e9f5; border-radius:8px; padding:10px 13px;
             overflow-x:auto; font-size:13px; margin:14px 0 10px;
             font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  pre.call .prompt {{ color:#7f8cff; user-select:none; }}
  p.desc {{ color:var(--muted); margin:0 0 14px; max-width:74ch; }}
  .render {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px;
             padding:10px; background:#fff; }}
  .render table {{ border-collapse:collapse; font-size:12.5px;
                   font-variant-numeric:tabular-nums; }}
  .render th, .render td {{ padding:4px 9px; border:1px solid #eef0f6;
                            text-align:right; white-space:nowrap; }}
  .render th {{ background:#f3f4fb; color:#3a3d4d; font-weight:600; }}
  .render th.blank {{ background:#fff; }}
  footer {{ text-align:center; color:var(--muted); font-size:13px; padding:30px; }}
  @media (max-width:820px) {{ .layout {{ grid-template-columns:1fr; }} nav {{ position:static; }} }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <h1>bunker-stats · Notebook UX</h1>
  <p>Every helper in <code>bunker_stats.notebook</code>, run on one messy sample
     table and rendered exactly as it appears in a notebook. Numbers come from
     the Rust kernels; the layer only validates inputs and handles NaN/∞.</p>
  <div class="install">pip install "bunker-stats-rs[notebook]"</div>
</div></header>

<div class="layout">
  <nav>
    {nav_section("report", "Reports")}
    {nav_section("columns", "Transforms")}
    {nav_section("style", "Stylers")}
  </nav>
  <main>
    {"".join(cards_html)}
    <footer>Generated by <code>examples/generate_gallery.py</code> ·
      bunker-stats notebook layer</footer>
  </main>
</div>
</body>
</html>"""

OUT.write_text(PAGE, encoding="utf-8")
print(f"Wrote {OUT} ({len(PAGE):,} bytes, {len(CARDS)} helpers)")
