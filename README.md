# ROGII Wellbore Geology Prediction

[![Autonomous Synthesis](https://img.shields.io/badge/Synthesized%20By-sovLoop%20(Stealth)-0A66C2?style=for-the-badge)](#how-this-pipeline-was-built)
[![Foundation LLM](https://img.shields.io/badge/Model-qwen3.5%3A9b-111827?style=for-the-badge)](#how-this-pipeline-was-built)
[![Validation](https://img.shields.io/badge/5--Fold%20OOF-RMSE%208.91536-059669?style=for-the-badge)](#verified-validation-benchmark)
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
| **Baseline 5-Fold OOF Score** | `9.64329` |
| **Champion 5-Fold OOF Score** | **`8.91536`** (`+0.72793` verified lift vs. baseline) |
| **Cross-Fold Stability (Std. Dev.)** | `±0.68590` across `5` strict folds |
| **Synthesis LLM** | `qwen3.5:9b` (via **sovLoop**) |

---

## Individual Model Benchmarks vs. Unrestricted Ensemble

Every individual model inside the unrestricted ensemble is benchmarked independently for standalone validation accuracy, training time, and inference latency—so production engineering teams can either deploy the **full unrestricted SOTA ensemble** or select the **fastest / best solo model** (`SELECTED_MODEL_KEY`) for strict SLA budgets:

| Model Family (`key`) | Standalone Val Score | Mean Fold Fit | Mean Fold Inference | Ensemble Weight | Deployment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CV-Tuned L2 Ridge Linear** (`ridge_l2`) | `8.90211 ± 0.66978` | `278.0 ms` | `11.66 ms` | `91.7%` | `Best Solo + Fastest SLA` |
| **CatBoost Symmetric** (`catboost`) | `14.94519 ± 1.90105` | `21658.8 ms` | `16.15 ms` | `0.5%` | `Standby Candidate` |
| **Optuna LightGBM** (`lgbm`) | `15.34221 ± 3.90508` | `8142.7 ms` | `25.93 ms` | `4.3%` | `Active Ensemble Member` |
| **Optuna XGBoost** (`xgb`) | `20.25758 ± 7.22213` | `16605.4 ms` | `21.99 ms` | `3.4%` | `Active Ensemble Member` |

---

## Autonomous Build Ledger (Commit-by-Commit Scientific Progression)

Every row below corresponds to a verified out-of-fold promotion recorded as an **isolated single-module commit** in `git log`:

| Iteration | Module Evolved | Active Synthesis Operations | 5-Fold OOF `RMSE` | Lift | Engineering Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0** | `scaffold` | Baseline initialization | `9.64329` | `baseline` | Deterministic 5-fold leak-free baseline anchor |
| **Gen 1** | `pipeline/models.py` (`neural_sequence_stream`) | `view_feat:plr_periodic_embeddings`, `tabm_periodic_mlp`, `realmlp_tuned_head`, `alignment_2d_convnext_unet`, `view_obj:kaplan_meier_target` | `9.56025` | `+0.08304` | Deploying the 2D Alignment ConvNeXt U-Net captures deep spatial correlations in wellbore sequences to handle heavy-tailed feature distributions, while... |
| **Gen 2** | `pipeline/postprocess.py` | `oof_residual_ridge` | `9.55794` | `+0.00231` | Applying fold-safe linear residual skip-correction on the OOF residuals will reduce RMSE below 6.1596 by modeling and removing systematic bias detected... |
| **Gen 3** | `pipeline/features.py` | `synth:null_co:pf_lik5_y:pf_lik8_y`, `synth:null_co:pf_lik5_y:pf_lik5_delta`, `synth:groupby_z:MD:pf_mean_y`, `synth:groupby_rank:MD:pf_mean_y`, `cat_groupby_stats` | `9.46518` | `+0.09276` | Encoding missingness in pf_lik* columns alongside MD-stratified group statistics will capture the 1.47x residual multiplicative bias by converting... |
| **Gen 4** | `pipeline/models.py` (`neural_sequence_stream`) | `view_feat:plr_periodic_embeddings`, `tabm_periodic_mlp`, `realmlp_tuned_head`, `alignment_2d_convnext_unet`, `alignment_2d_resnet34d_unet`, `view_obj:quantile_rank_target` | `8.93313` | `+0.53205` | Robust 2D spatial alignment handles wellbore geometric shifts, while quantile-rank target transformation provides rank-based robustness against... |
| **Gen 5** | `pipeline/models.py` (`gbdt_trajectory_stream`) | `lgbm_optuna_tuned`, `catboost_tuned`, `xgb_optuna_tuned`, `ridge_l2_linear` | `8.93312` | `+0.00001` | Coordinate-ascent probe on `gbdt_trajectory_stream`: Test 4-model ensemble adding `CV-Tuned L2 Ridge Linear Ensemble Head` |
| **Gen 6** | `pipeline/models.py` (`linear_residual_stream`) | `view_obj:quantile_rank_target`, `view_feat:plr_periodic_embeddings` | `8.91536` | `+0.01775` | Applying Rank-Gauss target smoothing neutralizes the heavy-tail residual dispersion caused by pf_std and pf_vs_hmm_gap outliers, while periodic... |

---

## Modular Architecture & Repository Layout

```mermaid
flowchart LR
    DATA["Raw Data (81 predictors)"] --> SPLIT["5-Fold CV Split"]
    SPLIT --> PRE["1. Preprocessing: mean_impute"]
    PRE --> FEAT["2. Features: synth:null_co:pf_lik5_y:pf_lik8_y, synth:null_co:pf_lik5_y:pf_lik5_delta, synth:groupby_z:MD:pf_mean_y, synth:groupby_rank:MD:pf_mean_y, +1 more"]
    FEAT --> OBJ["3. Objective: standard_task_loss"]
    OBJ --> M0["4a. Primary Ensemble: CV-Tuned L2 Ridge Linear (92%) + CatBoost Symmetric (1%) + Optuna LightGBM (4%) + Optuna XGBoost (3%)"]
    M0 --> FUSE["Quantile-CDF Fusion"]
    OBJ --> S1["4b. gbdt_trajectory_stream: lgbm_optuna_tuned, catboost_tuned, xgb_optuna_tuned, +1 more"]
    S1 --> FUSE
    OBJ --> S2["4b. linear_residual_stream: view_obj:quantile_rank_target, view_feat:plr_periodic_embeddings"]
    S2 --> FUSE
    OBJ --> S3["4b. neural_sequence_stream: view_feat:plr_periodic_embeddings, tabm_periodic_mlp, realmlp_tuned_head, +3 more"]
    S3 --> FUSE
    FUSE --> POST["5. Post-Processing: oof_residual_ridge"]
    POST --> OUT["5-Fold OOF & Holdout (RMSE = 8.91536)"]
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
