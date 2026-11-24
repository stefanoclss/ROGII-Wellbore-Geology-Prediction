"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: synth:null_co:pf_lik5_y:pf_lik8_y, synth:null_co:pf_lik5_y:pf_lik5_delta, synth:groupby_z:MD:pf_mean_y, synth:groupby_rank:MD:pf_mean_y, cat_groupby_stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('synth:null_co:pf_lik5_y:pf_lik8_y', 'synth:null_co:pf_lik5_y:pf_lik5_delta', 'synth:groupby_z:MD:pf_mean_y', 'synth:groupby_rank:MD:pf_mean_y', 'cat_groupby_stats')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['prefix_anchor_last', 'prefix_anchor_max', 'pf_mean_y', 'cnn2d_map_y', 'cnn2d_uproj_y', 'ref_axis_max']


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

    # Row-wise summary moments across predictors
    if fallback_cols:
        out['row_numeric_mean'] = out[fallback_cols].mean(axis=1)
        out['row_numeric_std'] = out[fallback_cols].std(axis=1).fillna(0.0)
    if len(fallback_cols) >= 2:
        _rk = out[fallback_cols[0]].rank(method='first')
        _q = pd.qcut(_rk, q=4, labels=False, duplicates='drop')
        _g = out.groupby(_q)[fallback_cols[1]].transform('mean')
        out['cohort_ctx_mean'] = _g
        out['cohort_ctx_resid'] = out[fallback_cols[1]] - _g
    if 'MD' in out.columns and 'pf_mean_y' in out.columns:
        _grp = out['pf_mean_y'].round(3)
        _gmu = out.groupby(_grp)['MD'].transform('mean')
        _gsd = out.groupby(_grp)['MD'].transform('std').fillna(1.0)
        out['MD_groupby_z_pf_mean_y'] = ((out['MD'] - _gmu) / _gsd.clip(lower=1e-2)).clip(-10, 10)
    if 'MD' in out.columns and 'pf_mean_y' in out.columns:
        _grp = out['pf_mean_y'].round(3)
        _gmu = out.groupby(_grp)['MD'].transform('mean')
        _gsd = out.groupby(_grp)['MD'].transform('std').fillna(1.0)
        out['MD_groupby_rank_pf_mean_y'] = ((out['MD'] - _gmu) / _gsd.clip(lower=1e-2)).clip(-10, 10)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
