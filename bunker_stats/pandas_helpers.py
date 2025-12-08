from __future__ import annotations

import numpy as np
import pandas as pd

from . import bunker_stats_rs


def demean_style(
    df: pd.DataFrame,
    column: str,
    *,
    above_color: str = "#c8e6c9",
    below_color: str = "#ffcdd2",
    zero_color: str = "#e0e0e0",
    add_suffix: str = "_demeaned",
) -> pd.io.formats.style.Styler:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    col = df[column].to_numpy(dtype="float64")
    demeaned, signs = bunker_stats_rs.demean_with_signs_np(col)
    demeaned = np.asarray(demeaned)
    signs = np.asarray(signs)

    demeaned_col = f"{column}{add_suffix}"
    out = df.copy()
    out[demeaned_col] = demeaned

    def _color_series(series: pd.Series, signs_: np.ndarray):
        styles = []
        for v, s in zip(series.to_numpy(), signs_):
            if pd.isna(v):
                styles.append("")
            elif s > 0:
                styles.append(f"background-color: {above_color}")
            elif s < 0:
                styles.append(f"background-color: {below_color}")
            else:
                styles.append(f"background-color: {zero_color}")
        return styles

    styler = out.style.apply(
        _color_series,
        axis=0,
        subset=[demeaned_col],
        signs_=signs,
    )
    return styler


def zscore_style(
    df: pd.DataFrame,
    column: str,
    *,
    threshold: float = 2.0,
    high_color: str = "#ffcc80",
    low_color: str = "#bbdefb",
    zero_color: str = "#f5f5f5",
    add_suffix: str = "_zscore",
) -> pd.io.formats.style.Styler:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    col = df[column].to_numpy(dtype="float64")
    z_arr = np.asarray(bunker_stats_rs.zscore_np(col))

    z_col = f"{column}{add_suffix}"
    out = df.copy()
    out[z_col] = z_arr

    def _style(series: pd.Series):
        vals = series.to_numpy()
        styles = []
        for z in vals:
            if pd.isna(z):
                styles.append("")
                continue
            if abs(z) >= threshold:
                if z > 0:
                    styles.append(f"background-color: {high_color}")
                else:
                    styles.append(f"background-color: {low_color}")
            else:
                styles.append(f"background-color: {zero_color}")
        return styles

    styler = out.style.apply(_style, axis=0, subset=[z_col])
    return styler


def iqr_outlier_style(
    df: pd.DataFrame,
    column: str,
    *,
    k: float = 1.5,
    outlier_color: str = "#ff8a80",
    normal_color: str = "",
) -> pd.io.formats.style.Styler:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    col = df[column].to_numpy(dtype="float64")
    mask = np.asarray(bunker_stats_rs.iqr_outliers_np(col, float(k)))

    def _style(series: pd.Series, mask_: np.ndarray):
        styles = []
        for is_out in mask_:
            styles.append(f"background-color: {outlier_color}" if is_out else normal_color)
        return styles

    styler = df.style.apply(_style, axis=0, subset=[column], mask_=mask)
    return styler


def corr_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    cmap: str = "coolwarm",
) -> pd.io.formats.style.Styler:
    if columns is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = columns

    if not cols:
        raise ValueError("No numeric columns to compute correlation on.")

    x = df[cols].to_numpy(dtype="float64")
    corr_mat = np.asarray(bunker_stats_rs.corr_matrix_np(x))
    corr_df = pd.DataFrame(corr_mat, index=cols, columns=cols)

    styler = corr_df.style.background_gradient(cmap=cmap)
    return styler


def robust_scale_column(
    df: pd.DataFrame,
    column: str,
    *,
    scale_factor: float = 1.4826,
    add_suffix: str = "_robust",
) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    col = df[column].to_numpy(dtype="float64")
    scaled, med, mad = bunker_stats_rs.robust_scale_np(col, float(scale_factor))
    out = df.copy()
    out[f"{column}{add_suffix}"] = np.asarray(scaled)
    return out
