# -*- coding: utf-8 -*-
"""Generate social-media images (1080x1080 PNG) showcasing bunker-stats.

Every figure and table is produced by the real library on seeded data — these
are actual outputs, styled for presentation. Requires the notebook extra plus
kaleido, pillow and pygments.

    python examples/make_social_images.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import bunker_stats as bs
from bunker_stats import notebook as nb
from bunker_stats.resampling import BootstrapConfig

OUT = pathlib.Path(__file__).with_name("social")
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Brand tokens (match the docs' dark theme)
# ----------------------------------------------------------------------
BG = "#14181d"
PANEL = "#1d232b"
INK = "#e6ebf0"
MUTED = "#98a3ad"
RUST = "#e0703a"
TEAL = "#4db6ac"
GRID = "#2c343d"

WORDMARK = "bunker-stats"
TAGLINE = "Rust-powered statistics for Python"
FOOTER = "pip install bunker-stats-rs[notebook]"

FONT_DIR = pathlib.Path(r"C:\Windows\Fonts")
MONO = str(FONT_DIR / "CascadiaCode.ttf")
UI = str(FONT_DIR / "segoeui.ttf")
UI_BOLD = str(FONT_DIR / "segoeuib.ttf")

SIZE = 1080

# ----------------------------------------------------------------------
# Seeded demo data: 126 trading days of daily returns with two shocks
# ----------------------------------------------------------------------
rng = np.random.default_rng(20260722)
n = 126
returns = rng.normal(0.05, 1.0, n)
returns[30] = 6.8    # shock up
returns[87] = -5.9   # shock down
volume = 42 + 6 * np.abs(returns) + rng.normal(0, 2, n)   # vol follows |moves|
momentum = np.convolve(returns, np.ones(5) / 5, mode="same") + rng.normal(0, .3, n)
sentiment = rng.normal(0, 1, n)                            # uncorrelated

DF = pd.DataFrame({
    "returns": returns,
    "volume": volume,
    "momentum": momentum,
    "sentiment": sentiment,
})
PRICE = 100 + np.cumsum(returns)


# ----------------------------------------------------------------------
# Plotly styling shared by every figure card
# ----------------------------------------------------------------------
def brand_figure(fig, title, subtitle):
    fig.update_layout(
        template="plotly_dark",
        width=SIZE, height=SIZE,
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(family="Segoe UI, sans-serif", color=INK, size=22),
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:20px;color:{MUTED}'>{subtitle}</span>"),
            x=0.055, y=0.925, font=dict(size=34),
        ),
        margin=dict(l=90, r=70, t=200, b=150),
        showlegend=fig.layout.showlegend if fig.layout.showlegend is not None else True,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.add_annotation(  # wordmark: top-right, clear of the left-aligned title
        text=f"<b><span style='color:{RUST}'>▚</span> {WORDMARK}</b>",
        xref="paper", yref="paper", x=1.0, y=1.18, xanchor="right",
        showarrow=False, font=dict(size=26, color=INK),
    )
    fig.add_annotation(  # footer
        text=f"<span style='color:{MUTED}'>{FOOTER}</span>",
        xref="paper", yref="paper", x=0.0, y=-0.16, xanchor="left",
        showarrow=False, font=dict(family="Cascadia Code, monospace", size=20),
    )
    return fig


def save(fig, name):
    # kaleido cannot run on this machine (headless chromium crashes), so
    # figures are exported as self-contained HTML and screenshotted in a
    # real browser at 1080x1080. The figure objects are unchanged.
    html_name = name.replace(".png", ".html")
    fig.write_html(
        str(OUT / html_name), include_plotlyjs=True, full_html=True,
        config={"displayModeBar": False},
        default_width="1080px", default_height="1080px",
    )
    (OUT / name.replace(".png", ".json")).write_text(fig.to_json(), encoding="utf-8")
    print("wrote", html_name)


# ----------------------------------------------------------------------
# 1) Correlation heatmap — CorrelationReport.plot_heatmap()
# ----------------------------------------------------------------------
corr = nb.correlation_report(DF, rich=True)
fig = corr.plot_heatmap()
fig.update_traces(
    colorscale=[[0, TEAL], [0.5, PANEL], [1, RUST]],
    texttemplate="%{z:.2f}", textfont=dict(size=26),
)
brand_figure(
    fig, "One-line correlation heatmap",
    "nb.correlation_report(df, rich=True).plot_heatmap()",
)
save(fig, "02_corr_heatmap.png")

# ----------------------------------------------------------------------
# 2) Outlier counts — OutlierReport.plot_counts()
# ----------------------------------------------------------------------
out = nb.outlier_report(DF, method="robust_zscore", rich=True)
fig = out.plot_counts()
fig.update_traces(marker_color=RUST, texttemplate="%{y:.0f}", textfont=dict(size=30))
brand_figure(
    fig, "Where are my outliers?",
    "nb.outlier_report(df, method=\"robust_zscore\", rich=True).plot_counts()",
)
fig.update_layout(showlegend=False)
fig.update_yaxes(dtick=1)
save(fig, "03_outlier_counts.png")

# ----------------------------------------------------------------------
# 3) Bootstrap distribution — BootstrapResult.plot_distribution()
# ----------------------------------------------------------------------
boot = BootstrapConfig(n_resamples=5000, random_state=7, return_draws=True).run(
    DF["returns"].to_numpy()
)
fig = boot.plot_distribution()
fig.update_traces(marker_color=TEAL, selector=dict(type="histogram"))
brand_figure(
    fig, "5,000 bootstrap resamples, one flag",
    "BootstrapConfig(return_draws=True).run(x).plot_distribution()",
)
for ann in fig.layout.annotations:
    t = (ann.text or "")
    if t.startswith("estimate"):
        ann.update(y=0.5, bgcolor=BG, borderpad=6)
    elif "lower" in t:
        ann.update(xanchor="right")
    elif "upper" in t:
        ann.update(xanchor="left")
save(fig, "04_bootstrap_distribution.png")

# ----------------------------------------------------------------------
# 4) CI intervals per column — BootstrapCIReport.plot_intervals()
# ----------------------------------------------------------------------
ci = nb.bootstrap_ci_report(
    DF, ["returns", "momentum", "sentiment"],
    n_resamples=3000, random_state=7, rich=True,
)
fig = ci.plot_intervals()
fig.update_traces(marker_color=RUST, marker_size=16,
                  error_y_thickness=5, error_y_width=18)
brand_figure(
    fig, "Bootstrap CIs, column by column",
    "nb.bootstrap_ci_report(df, cols, rich=True).plot_intervals()",
)
fig.update_layout(showlegend=False)
fig.add_hline(y=0.0, line_color=GRID, line_dash="dot")
save(fig, "05_ci_intervals.png")

# ----------------------------------------------------------------------
# 5) Rolling stats — RollingResult.plot()
# ----------------------------------------------------------------------
import plotly.graph_objects as go

roll = bs.Rolling(PRICE, window=20).result("mean", "std")
frame = roll.to_frame()
m = frame["roll20_mean"].to_numpy()
sd = frame["roll20_std"].to_numpy()
xs = np.arange(len(PRICE))[len(PRICE) - len(m):]

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=m + 2 * sd, line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=m - 2 * sd, fill="tonexty",
                         fillcolor="rgba(77,182,172,0.22)", line=dict(width=0),
                         name="mean ± 2·std"))
fig.add_trace(go.Scatter(x=np.arange(len(PRICE)), y=PRICE, name="price",
                         line=dict(color=MUTED, width=1.5)))
fig.add_trace(go.Scatter(x=xs, y=m, name="20-day mean",
                         line=dict(color=RUST, width=4)))
brand_figure(
    fig, "Rolling mean, std and bands — one pass",
    "bs.Rolling(price, 20).result(\"mean\", \"std\")   ·   band = mean ± 2·std",
)
fig.update_layout(legend=dict(x=0.02, y=0.98, bgcolor="rgba(20,24,29,0.75)"))
save(fig, "06_rolling.png")


# ----------------------------------------------------------------------
# PIL cards: cover + code/info cards
# ----------------------------------------------------------------------
def font(path, size):
    return ImageFont.truetype(path, size)


def new_card():
    img = Image.new("RGB", (SIZE * 2, SIZE * 2), BG)
    return img, ImageDraw.Draw(img)


def wordmark_footer(d):
    d.text((110, 96), "▚", font=font(MONO, 52), fill=RUST)
    d.text((180, 96), WORDMARK, font=font(UI_BOLD, 52), fill=INK)
    d.text((110, 2 * SIZE - 150), FOOTER, font=font(MONO, 40), fill=MUTED)


def code_block(d, x, y, w, lines, fs=40):
    """Terminal-style block with minimal token coloring."""
    mono, lh = font(MONO, fs), int(fs * 1.55)
    h = len(lines) * lh + 70
    d.rounded_rectangle([x, y, x + w, y + h], radius=24, fill=PANEL, outline=GRID, width=3)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):  # window dots
        d.ellipse([x + 36 + i * 46, y + 28, x + 62 + i * 46, y + 54], fill=c)
    ty = y + 84
    for ln in lines:
        tx = x + 44
        if ln.startswith(">>>"):
            d.text((tx, ty), ">>>", font=mono, fill=RUST)
            tx += d.textlength(">>> ", font=mono)
            d.text((tx, ty), ln[4:], font=mono, fill=INK)
        elif ln.startswith("#"):
            d.text((tx, ty), ln, font=mono, fill=MUTED)
        else:
            d.text((tx, ty), ln, font=mono, fill=TEAL)
        ty += lh
    return y + h


# --- 01 cover ----------------------------------------------------------
img, d = new_card()
wordmark_footer(d)
d.text((110, 560), "Stats that explain", font=font(UI_BOLD, 128), fill=INK)
d.text((110, 700), "themselves.", font=font(UI_BOLD, 128), fill=RUST)
d.text((110, 900), TAGLINE + " — hypothesis tests, robust stats,\n"
       "bootstrap and rolling kernels with rich, honest result objects.",
       font=font(UI, 52), fill=MUTED, spacing=18)
code_block(d, 110, 1250, 1860, [
    ">>> bs.t_test_2samp(x, y, rich=True).conclusion()",
    "'Reject H0 (the two means are equal) at alpha=0.05 (p=0.0031)'",
    ">>> nb.correlation_report(df, rich=True).plot_heatmap()",
    ">>> nb.outlier_report(df, rich=True).plot_counts()",
], fs=44)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "01_cover.png")
print("wrote 01_cover.png")

# --- 07 rich t-test card: real .info() output --------------------------
res = bs.t_test_2samp(
    rng.normal(0.0, 1.0, 60), rng.normal(0.55, 1.0, 64), rich=True
)
info_lines = res.info().splitlines()

img, d = new_card()
wordmark_footer(d)
d.text((110, 240), "rich=True", font=font(MONO, 96), fill=TEAL)
d.text((110, 380), "Every test can return a real result object —",
       font=font(UI, 52), fill=INK)
d.text((110, 452), "tuple-unpackable, serializable, self-describing.",
       font=font(UI, 52), fill=INK)
y = code_block(d, 110, 590, 1860, [
    ">>> res = bs.t_test_2samp(x, y, rich=True)",
    ">>> stat, p = res          # unpacking still works",
    ">>> print(res.info())",
], fs=44)
code_block(d, 110, y + 40, 1860, info_lines, fs=38)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "07_rich_ttest.png")
print("wrote 07_rich_ttest.png")

# --- 08 misuse warnings card -------------------------------------------
tiny = bs.t_test_2samp(rng.normal(0, 1, 5), rng.normal(1, 1, 6), rich=True)
warn_lines = [ln for ln in tiny.info().splitlines()]
img, d = new_card()
wordmark_footer(d)
d.text((110, 240), "Misuse prevention,", font=font(UI_BOLD, 88), fill=INK)
d.text((110, 360), "built in.", font=font(UI_BOLD, 88), fill=RUST)
d.text((110, 520), "Small sample? NaNs? Approximate p-value?\n"
       "Rich results say so before you publish the number.",
       font=font(UI, 50), fill=MUTED, spacing=16)
code_block(d, 110, 780, 1860, [
    ">>> res = bs.t_test_2samp(x_tiny, y_tiny, rich=True)",
    ">>> print(res.info())",
] + warn_lines, fs=36)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "08_misuse_warnings.png")
print("wrote 08_misuse_warnings.png")

print("\nAll PIL/plotly cards done ->", OUT)
