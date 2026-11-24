"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: log1p_skewed
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('log1p_skewed',)
PRIORITY_SIGNAL_COLUMNS: list[str] = ['prefix_anchor_last', 'pf_mean_y', 'align_dp_med', 'align_fb_med', 'align_dp_tight', 'prefix_anchor_max']


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
    if 'prefix_anchor_last' in out.columns: out['prefix_anchor_last_log1p'] = np.sign(out['prefix_anchor_last']) * np.log1p(np.abs(out['prefix_anchor_last']))
    if 'pf_mean_y' in out.columns: out['pf_mean_y_log1p'] = np.sign(out['pf_mean_y']) * np.log1p(np.abs(out['pf_mean_y']))
    if 'align_dp_med' in out.columns: out['align_dp_med_log1p'] = np.sign(out['align_dp_med']) * np.log1p(np.abs(out['align_dp_med']))
    for _c in fallback_cols[:3]:
        if f'{_c}_log1p' not in out.columns:
            out[f'{_c}_log1p'] = np.sign(out[_c]) * np.log1p(np.abs(out[_c]))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
