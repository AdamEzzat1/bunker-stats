# -*- coding: utf-8 -*-
"""Table cards for the social set: outlier_style + robust_summary, via PIL.

Numbers and outlier masks come from the real library; PIL only draws them.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import bunker_stats as bs
from bunker_stats import notebook as nb

OUT = pathlib.Path(__file__).with_name("social")

BG, PANEL, INK, MUTED = "#14181d", "#1d232b", "#e6ebf0", "#98a3ad"
RUST, TEAL, GRID = "#e0703a", "#4db6ac", "#2c343d"
HILITE_TXT = "#14181d"
WORDMARK, FOOTER = "bunker-stats", "pip install bunker-stats[notebook]"

F = pathlib.Path(r"C:\Windows\Fonts")
MONO, UI, UI_BOLD = str(F / "CascadiaCode.ttf"), str(F / "segoeui.ttf"), str(F / "segoeuib.ttf")
SIZE = 1080

rng = np.random.default_rng(20260722)
n = 126
returns = rng.normal(0.05, 1.0, n)
returns[30] = 6.8
returns[87] = -5.9
volume = 42 + 6 * np.abs(returns) + rng.normal(0, 2, n)
momentum = np.convolve(returns, np.ones(5) / 5, mode="same") + rng.normal(0, .3, n)
sentiment = rng.normal(0, 1, n)
DF = pd.DataFrame({"returns": returns, "volume": volume,
                   "momentum": momentum, "sentiment": sentiment})


def font(path, size):
    return ImageFont.truetype(path, size)


def card_top(d, title_plain, title_accent, subtitle):
    d.text((110, 96), "▚", font=font(MONO, 52), fill=RUST)
    d.text((180, 96), WORDMARK, font=font(UI_BOLD, 52), fill=INK)
    d.text((110, 260), title_plain, font=font(UI_BOLD, 92), fill=INK)
    w = d.textlength(title_plain, font=font(UI_BOLD, 92))
    d.text((110 + w, 260), title_accent, font=font(UI_BOLD, 92), fill=RUST)
    d.text((110, 400), subtitle, font=font(MONO, 42), fill=MUTED)
    d.text((110, 2 * SIZE - 150), FOOTER, font=font(MONO, 40), fill=MUTED)


def draw_table(d, x, y, col_w, row_h, headers, rows, highlights, fs=38):
    mono_b, mono = font(MONO, fs), font(MONO, fs)
    ui_b = font(UI_BOLD, fs)
    width = sum(col_w)
    # header
    d.rounded_rectangle([x, y, x + width, y + row_h], radius=0, fill=PANEL)
    cx = x
    for h, w in zip(headers, col_w):
        d.text((cx + w - 24 - d.textlength(h, font=ui_b), y + row_h / 2 - fs / 2),
               h, font=ui_b, fill=MUTED)
        cx += w
    # body
    for r, row in enumerate(rows):
        ry = y + (r + 1) * row_h
        cx = x
        for c, (val, w) in enumerate(zip(row, col_w)):
            hit = (r, c) in highlights
            if hit:
                d.rectangle([cx + 6, ry + 5, cx + w - 6, ry + row_h - 5], fill=RUST)
            color = HILITE_TXT if hit else (INK if c == 0 else TEAL)
            f_use = mono_b if hit else mono
            d.text((cx + w - 24 - d.textlength(val, font=f_use), ry + row_h / 2 - fs / 2),
                   val, font=f_use, fill=color)
            cx += w
        d.line([x, ry, x + width, ry], fill=GRID, width=2)
    d.line([x, y + (len(rows) + 1) * row_h, x + width, y + (len(rows) + 1) * row_h],
           fill=GRID, width=2)


# --- 09: outlier_style ---------------------------------------------------
img = Image.new("RGB", (SIZE * 2, SIZE * 2), BG)
d = ImageDraw.Draw(img)
card_top(d, "Bad rows, ", "highlighted.",
         'nb.outlier_style(df)   # every numeric column at once')

cols = ["returns", "volume", "momentum", "sentiment"]
masks = {c: np.asarray(bs.iqr_outliers(DF[c].to_numpy(), 1.5)) for c in cols}
window = list(range(27, 34)) + [86, 87, 88]
headers = ["day"] + cols
rows, highlights = [], set()
for r, i in enumerate(window):
    row = [str(i)]
    for c, colname in enumerate(cols):
        row.append(f"{DF[colname].iloc[i]:.2f}")
        if masks[colname][i]:
            highlights.add((r, c + 1))
    rows.append(row)

draw_table(d, 110, 560, [220, 400, 400, 400, 400], 118, headers, rows, highlights, fs=40)
d.text((110, 560 + 11 * 118 + 40),
       "IQR fences from the Rust kernels — the day-30 spike and day-87 crash\n"
       "light up in every affected column. NaN cells are never flagged.",
       font=font(UI, 44), fill=MUTED, spacing=14)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "09_outlier_table.png")
print("wrote 09_outlier_table.png")

# --- 10: robust_summary --------------------------------------------------
img = Image.new("RGB", (SIZE * 2, SIZE * 2), BG)
d = ImageDraw.Draw(img)
card_top(d, "describe(), but ", "robust.",
         "nb.robust_summary(df)   # median, MAD, IQR, Qn, trimmed mean")

summ = nb.robust_summary(DF)
keep = ["n", "mean", "std", "median", "mad", "iqr", "skew"]
headers = ["column"] + keep
rows = []
for name, row in summ.iterrows():
    cells = [str(name)]
    for k in keep:
        v = row[k]
        cells.append(f"{v:.0f}" if k == "n" else f"{v:.2f}")
    rows.append(cells)

draw_table(d, 110, 600, [340, 170, 240, 220, 250, 220, 220, 220], 130,
           headers, rows, set(), fs=40)
d.text((110, 600 + 6 * 130 + 60),
       "The mean and std feel the day-30 and day-87 shocks; the median,\n"
       "MAD and IQR barely move. That gap IS the outlier story.",
       font=font(UI, 44), fill=MUTED, spacing=14)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "10_robust_summary.png")
print("wrote 10_robust_summary.png")
