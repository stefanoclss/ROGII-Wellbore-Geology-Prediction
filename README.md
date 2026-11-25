# ROGII Wellbore Geology Prediction

[![Autonomous Synthesis](https://img.shields.io/badge/Synthesized%20By-sovLoop%20(Stealth)-0A66C2?style=for-the-badge)](#how-this-pipeline-was-built)
[![Foundation LLM](https://img.shields.io/badge/Model-qwen3.5%3A9b-111827?style=for-the-badge)](#how-this-pipeline-was-built)
[![Validation](https://img.shields.io/badge/5--Fold%20OOF-RMSE%208.20575-059669?style=for-the-badge)](#verified-validation-benchmark)
[![Security](https://img.shields.io/badge/Zero--Pickle-CWE--502%20Compliant-374151?style=for-the-badge)](#governance--production-guarantees)
[![Kaggle Public Notebook](https://img.shields.io/badge/Kaggle%20Notebook-Public%20Kernel-20BEFF?style=for-the-badge)](https://www.kaggle.com/code/stefanoclss/rogii-wellbore-geology-prediction)
[![Kaggle Challenge](https://img.shields.io/badge/Challenge-Benchmark-0284C7?style=for-the-badge)](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

> **Autonomous End-to-End Machine Learning Engineering by sovLoop**
> From raw tabular schema forensics to interpretable `preprocessing(df)` & `feature_engineering(df)` pipelines, Optuna-tuned multi-family GBDT ensembles, and zero-pickle production packaging—evolved step-by-step using **`qwen3.5:9b`**.

---

## Executive Overview

Predict `TVT` (regression) across `81` structured predictors on `rogii-wellbore-geology-prediction` under strict leak-free 5-fold out-of-fold validation to minimize `RMSE`.

Production data science teams need pipelines that are not only accurate, but **readable, explainable, and maintainable**. Every transformation in this repository is organized by pipeline topology stage (`preprocessing`, `feature_engineering`, `objective`, `models`, `postprocess`) and validated incrementally via **sovLoop** (operating in private stealth mode).

---

## How This Pipeline Was Built

1. **Simple Baseline -> Gradual Verified Evolution**: Starting from a clean baseline, **sovLoop** (guided by `qwen3.5:9b`) evolved one stage at a time—adding explicit column hygiene, domain features, leak-free target formulation, and Optuna hyperparameter optimization.
2. **Strict Out-of-Fold Scientific Gatekeeping**: Candidate modifications to preprocessing, feature engineering, estimator hyperparameters, and post-processing were evaluated across **100% of 5-fold out-of-fold validation rows**. Any change that failed to improve CV was rejected.
3. **Atomic Single-Module Git Lineage**: After initial scaffolding, **each commit in this repository modifies strictly one atomic stage file** (`pipeline/models.py`, `pipeline/features.py`, `pipeline/preprocess.py`, `pipeline/objective.py`, or `pipeline/postprocess.py`). Inspect `git log -p` to audit every single lift.
4. **Single-Notebook End-to-End Reproducibility (Zero-Pickle / No Git LFS)**: Place raw tables in `data/` and run **`pipeline.ipynb`** top-to-bottom to reproduce visual EDA (`eda_pipeline.py`), K-Fold LightGBM/Optuna training (`training_pipeline.py` -> `artifacts/model_artifact.json`), and pure artifact holdout inferencing (`inference_pipeline.py` -> `submission.csv`).

---

## Verified Validation Benchmark

| Benchmark Dimension | Verified Result |
| :--- | :--- |
| **Challenge / Dataset** | **ROGII Wellbore Geology Prediction** (`rogii-wellbore-geology-prediction`) |
| **Target Variable (`y`)** | `TVT` (`regression`) |
| **Evaluation KPI** | `RMSE` (Lower is better) |
| **Baseline 5-Fold OOF Score** | `9.47203` |
| **Champion 5-Fold OOF Score** | **`8.20575`** (`+1.26628` verified lift vs. baseline) |
| **Cross-Fold Stability (Std. Dev.)** | `±0.61196` across `5` strict folds |
| **Synthesis LLM** | `qwen3.5:9b` (via **sovLoop**) |

---

## Individual Model Benchmarks vs. Unrestricted Ensemble

Every individual model inside the unrestricted ensemble is benchmarked independently for standalone validation accuracy, training time, and inference latency—so production engineering teams can either deploy the **full unrestricted SOTA ensemble** or select the **fastest / best solo model** (`SELECTED_MODEL_KEY`) for strict SLA budgets:

| Model Family (`key`) | Standalone Val Score | Mean Fold Fit | Mean Fold Inference | Ensemble Weight | Deployment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost Symmetric** (`catboost`) | `8.17634 ± 0.62119` | `4935.0 ms` | `21.65 ms` | `31.3%` | `Best Solo Accuracy` |
| **Optuna XGBoost** (`xgb`) | `8.18239 ± 0.58464` | `9721.5 ms` | `28.20 ms` | `38.1%` | `Active Ensemble Member` |
| **Optuna LightGBM** (`lgbm`) | `8.22728 ± 0.51483` | `5698.7 ms` | `23.21 ms` | `5.3%` | `Active Ensemble Member` |
| **CV-Tuned L2 Ridge Linear** (`ridge_l2`) | `8.34897 ± 0.57744` | `383.9 ms` | `17.91 ms` | `25.3%` | `Fastest Low-Latency` |

---

## Autonomous Build Ledger (Commit-by-Commit Scientific Progression)

Every row below corresponds to a verified out-of-fold promotion recorded as an **isolated single-module commit** in `git log`:

| Iteration | Module Evolved | Active Synthesis Operations | 5-Fold OOF `RMSE` | Lift | Engineering Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0** | `scaffold` | Baseline initialization | `9.47203` | `baseline` | Deterministic 5-fold leak-free baseline anchor |
| **Gen 1** | `pipeline/models.py` (`gbdt_trajectory_stream`) | `lgbm_optuna_tuned`, `catboost_tuned`, `xgb_optuna_tuned`, `ridge_l2_linear` | `9.46048` | `+0.01156` | Coordinate-ascent probe on `gbdt_trajectory_stream`: Test 4-model ensemble adding `CV-Tuned L2 Ridge Linear Ensemble Head` |
| **Gen 2** | `pipeline/models.py` (`linear_residual_stream`) | `view_feat:row_summary_stats`, `ridge_l2_linear`, `lgbm_optuna_tuned`, `view_obj:standard_task_loss` | `9.45924` | `+0.00123` | Integrating an Optuna-tuned LightGBM head with standard loss and row summary statistics will capture non-linear interactions in skewed features (pf_std... |
| **Gen 3** | `pipeline/models.py` (`neural_sequence_stream`) | `view_feat:plr_periodic_embeddings`, `tabm_periodic_mlp`, `realmlp_tuned_head`, `alignment_2d_convnext_unet` | `8.62147` | `+0.83777` | Coordinate-ascent probe on `neural_sequence_stream`: Test 4-model ensemble adding `2D Alignment ConvNeXt-Tiny U-Net + BiGRU + 3-View TTA (GPU)` |
| **Gen 4** | `pipeline/objective.py` | `anchor_residual_target` | `8.30770` | `+0.31377` | Modeling the target as a residual from the anchor prefix while applying Huber loss will simultaneously leverage the informative null patterns of... |
| **Gen 5** | `pipeline/postprocess.py` | `oof_residual_ridge` | `8.30463` | `+0.00307` | Applying OOF-residual ridge correction to identity-passthrough predictions will reduce RMSE from 5.44 to <4.8 by linearly modeling the systematic... |
| **Gen 6** | `pipeline/models.py` (`neural_sequence_stream`) | `view_feat:plr_periodic_embeddings`, `tabm_periodic_mlp`, `realmlp_tuned_head`, `alignment_2d_convnext_unet`, `alignment_2d_resnet34d_unet` | `8.21180` | `+0.09284` | Coordinate-ascent probe on `neural_sequence_stream`: Test 5-model ensemble adding `2D Alignment ResNet34d U-Net + BiGRU + 3-View TTA (GPU)` |
| **Gen 7** | `pipeline/models.py` (`stream_1`) | `view_obj:log1p_skewed_target`, `view_feat:pairwise_diff_crosses`, `xgb_optuna_tuned` | `8.21174` | `+0.00006` | Spawn orthogonal signed-log1p target stream (XGBoost) to break plateau. |
| **Gen 8** | `pipeline/models.py` (`stream_1`) | `view_obj:log1p_skewed_target`, `view_feat:pairwise_diff_crosses`, `xgb_optuna_tuned`, `lgbm_optuna_tuned` | `8.21147` | `+0.00027` | Coordinate-ascent probe on `stream_1`: Test 4-model ensemble adding `Optuna-Tuned LightGBM (Bayesian TPE Hyperparameter Search)` |
| **Gen 9** | `pipeline/models.py` (`linear_residual_stream`) | `xgb_optuna_tuned`, `view_feat:pairwise_diff_crosses`, `ridge_l2_linear`, `view_obj:standard_task_loss` | `8.20575` | `+0.00572` | Replacing LGBM with XGBoost introduces histogram-based splitting and distinct leaf-weight regularization that better resolves the positive residual... |

---

## Modular Architecture & Repository Layout

```mermaid
flowchart LR
    DATA["Raw Data (81 predictors)"] --> SPLIT["5-Fold CV Split"]
    SPLIT --> PRE["1. Preprocessing: mean_impute"]
    PRE --> FEAT["2. Features: raw_passthrough"]
    FEAT --> OBJ["3. Objective: anchor_residual_target"]
    OBJ --> M0["4a. Primary Ensemble: CatBoost Symmetric (31%) + Optuna XGBoost (38%) + Optuna LightGBM (5%) + CV-Tuned L2 Ridge Linear (25%)"]
    M0 --> FUSE["Quantile-CDF Fusion"]
    OBJ --> S1["4b. gbdt_trajectory_stream: lgbm_optuna_tuned, catboost_tuned, xgb_optuna_tuned, +1 more"]
    S1 --> FUSE
    OBJ --> S2["4b. linear_residual_stream: xgb_optuna_tuned, view_feat:pairwise_diff_crosses, ridge_l2_linear, +1 more"]
    S2 --> FUSE
    OBJ --> S3["4b. neural_sequence_stream: view_feat:plr_periodic_embeddings, tabm_periodic_mlp, realmlp_tuned_head, +2 more"]
    S3 --> FUSE
    OBJ --> S4["4b. stream_1: view_obj:log1p_skewed_target, view_feat:pairwise_diff_crosses, xgb_optuna_tuned, +1 more"]
    S4 --> FUSE
    FUSE --> POST["5. Post-Processing: oof_residual_ridge"]
    POST --> OUT["5-Fold OOF & Holdout (RMSE = 8.20575)"]
```

| Deliverable | Role in Production Workflow |
| :--- | :--- |
| **`pipeline.ipynb`** | **Primary Topology Walkthrough Notebook**: explains every stage with visual charts and reproduces EDA, K-Fold training, and holdout scoring. |
| `eda.ipynb` / `eda_pipeline.py` | Schema diagnostics, missingness bar charts, and target distribution plots via `pipeline/eda.py`. |
| `train.ipynb` / `training_pipeline.py` | Leak-free K-Fold LightGBM + Optuna CV orchestrator exporting zero-pickle `artifacts/model_artifact.json` + `fold_*.lgb`. |
| `inference.ipynb` / `inference_pipeline.py` | Pure artifact-driven holdout scoring (`--artifact artifacts/model_artifact.json`) with zero re-training. |
| `pipeline/*.py` | Readable stage modules (`preprocessing(df)`, `feature_engineering(df)`, `models.py`) evolved one file per commit. |

---

## Governance & Production Guarantees

- **100% Validation Fold Integrity**: Imputation statistics, IQR bounds, Optuna inner splits, and target transforms fit strictly inside training folds.
- **Zero-Pickle Security (`CWE-502`)**: Models serialize to human-auditable LightGBM text/JSON (`fold_*.lgb`, `model_artifact.json`) — zero unsafe `.pkl` binaries.
- **Data-Scientist Readability**: Explicit column-by-column `pandas` transformations that any engineer can inspect, explain to stakeholders, and extend.

---

## Quickstart (1-Notebook Reproduction & CLI)

```bash
pip install -r requirements.txt

# Option A — Reproduce Training + Inferencing in ONE Notebook
jupyter execute pipeline.ipynb

# Option B — Modular CLI Orchestrators
python eda_pipeline.py --input data/train.csv
python training_pipeline.py --train data/train.csv --artifact-out artifacts/model_artifact.json --oof-out oof_predictions.csv
python inference_pipeline.py --artifact artifacts/model_artifact.json --test data/test.csv --output submission.csv
```

---

## About the Author

### Stefano Claes — AI Systems Architect

Working on **sovereign local-first AI, neuro-symbolic & agentic harnesses, and closed-loop recursive self-improvement**—pairing autonomous iteration with deterministic verification, zero-trust sandboxing, and human-readable engineering.

| Architectural Pillar | Standard |
| :--- | :--- |
| **Sovereign & Local-First AI** | Self-contained, air-gapped execution over open-weights models with zero external data exfiltration or cloud lock-in. |
| **Neuro-Symbolic Harnesses** | Deterministic analytical kernels, structured tool execution, and composable topologies constraining autonomous search. |
| **Recursive Verification** | Closed-loop iteration gated by strict out-of-sample evaluation and atomic single-module Git lineage. |
| **Zero-Trust & Artifact Security (`CWE-502`)** | Zero-pickle serialization, human-auditable Python modules, latency-vs-accuracy profiling, and reproducible standalone inference. |

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Stefano%20Claes-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/stefano-claes-266435140/) [![GitHub](https://img.shields.io/badge/GitHub-%40stefanoclss-111827?style=for-the-badge&logo=github&logoColor=white)](https://github.com/stefanoclss)

- **Author**: **Stefano Claes** (AI Systems Architect)
- **LinkedIn**: [https://www.linkedin.com/in/stefano-claes-266435140/](https://www.linkedin.com/in/stefano-claes-266435140/)
- **GitHub**: [https://github.com/stefanoclss](https://github.com/stefanoclss)
