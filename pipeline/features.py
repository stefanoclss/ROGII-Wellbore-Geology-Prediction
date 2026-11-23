"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: row_summary_stats, pairwise_diff_crosses, plr_periodic_embeddings, top_pair_ratios, log1p_skewed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('row_summary_stats', 'pairwise_diff_crosses', 'plr_periodic_embeddings', 'top_pair_ratios', 'log1p_skewed')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['pf_lik5_delta', 'ref_axis_range', 'prefix_anchor_range', 'prefix_slope_all', 'coupled_tail_dip', 'ref_axis_max']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    # Signed log1p compression for skewed continuous predictors
    if 'pf_lik5_delta' in out.columns: out['pf_lik5_delta_log1p'] = np.sign(out['pf_lik5_delta']) * np.log1p(np.abs(out['pf_lik5_delta']))
    if 'ref_axis_range' in out.columns: out['ref_axis_range_log1p'] = np.sign(out['ref_axis_range']) * np.log1p(np.abs(out['ref_axis_range']))
    if 'prefix_anchor_range' in out.columns: out['prefix_anchor_range_log1p'] = np.sign(out['prefix_anchor_range']) * np.log1p(np.abs(out['prefix_anchor_range']))
    for _c in fallback_cols[:3]:
        if f'{_c}_log1p' not in out.columns:
            out[f'{_c}_log1p'] = np.sign(out[_c]) * np.log1p(np.abs(out[_c]))
    # Safe domain ratios between top numeric predictors
    if 'pf_lik5_delta' in out.columns and 'ref_axis_range' in out.columns:
        _den = out['ref_axis_range'].where(out['ref_axis_range'].abs() >= 0.15, 0.15)
        out['pf_lik5_delta_div_ref_axis_range'] = (out['pf_lik5_delta'] / _den).clip(-10.0, 10.0)
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        _rname = f'{_a}_div_{_b}'
        if _rname not in out.columns:
            _den = out[_b].where(out[_b].abs() >= 0.15, 0.15)
            out[_rname] = (out[_a] / _den).clip(-10.0, 10.0)
    # Pairwise differences & absolute discrepancies
    if len(fallback_cols) >= 2:
        _a, _b = fallback_cols[0], fallback_cols[1]
        out[f'{_a}_minus_{_b}'] = out[_a] - out[_b]
        out[f'{_a}_absdiff_{_b}'] = (out[_a] - out[_b]).abs()
    # Smooth periodic numerical embeddings (sin/cos harmonics)
    for _c in fallback_cols[:3]:
        _scaled = out[_c].clip(-10.0, 10.0) * (2.0 * np.pi * 0.20)
        out[f'{_c}_sin'] = np.sin(_scaled)
        out[f'{_c}_cos'] = np.cos(_scaled)
    # Row-wise summary moments across predictors
    if fallback_cols:
        out['row_numeric_mean'] = out[fallback_cols].mean(axis=1)
        out['row_numeric_std'] = out[fallback_cols].std(axis=1).fillna(0.0)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
