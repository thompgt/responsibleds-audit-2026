# Cleaned Model Pipeline — Detailed Workflow

This document summarizes the end-to-end workflow implemented in the `cleaned_model_pipeline.ipynb` notebook. It describes each logical stage, the intent behind it, key inputs and outputs, and the evaluation and fairness checks used to validate models. No code is included; this is a conceptual and operational guide for engineers and reviewers.

## Purpose and Scope
- Objective: build a robust, auditable pipeline to predict football transfer fees using transfer records and FIFA rating datasets.
- Scope: data ingestion, cleaning, feature engineering (including FIFA integration), multiple modeling experiments, fairness-aware preprocessing and modeling, diagnostics for Simpson's paradox, hierarchical modeling, and explainability.

## 1. Environment Setup & Data Loading
- Purpose: ensure dependencies and load raw data assets in a reproducible way.
- Inputs: compressed transfer records and compressed FIFA ratings files located under `data/`.
- Key actions: load transfer dataset, define normalization helper(s), and inspect schemas to confirm column availability.
- Output: raw transfer DataFrame accessible to subsequent steps.

## 2. Initial Data Cleaning
- Purpose: remove obviously invalid records and harmonize categorical values for consistency.
- Actions: extract and normalize season year, standardize position labels, filter out rows with invalid ages (e.g., Age == 0), and compute fee and market-value columns in consistent units (millions).
- Output: a cleaned transfer dataset with consistent types and baseline aggregate season statistics saved for later reference.

## 3. Basic Feature Engineering
- Purpose: create straightforward, deterministic features to enable reliable joins and downstream modeling.
- Actions: focus on a recent period (2015–2018), extract last names for fuzzy joins, normalize text fields (player names, teams), and create compact subsets for modeling.
- Output: a candidate modeling set limited to target seasons with normalized join keys.

## 4. FIFA Ratings Integration (Advanced Feature Engineering)
- Purpose: augment transfer records with player ability and potential proxies from FIFA datasets to improve predictive signal.
- Actions: load FIFA player tables across multiple years, extract relevant fields (overall, potential, club, nationality), normalize names and clubs, create per-season lookup tables, and select best-scoring records as primary matches.
- Fallbacks: retain both club-level and lastname-season-level lookups to increase match coverage.
- Output: a FIFA-enriched lookup table and merged transfers dataset containing `overall` and `potential` where available.

## 5. Dataset Merging and Fallback Recovery
- Purpose: robustly match transfers to FIFA lookups using a tiered fallback strategy to maximize coverage while limiting false matches.
- Matching hierarchy: Team_from join → Team_to join → broader Lastname+Season lookup.
- Outcome: consolidated dataset where FIFA attributes are filled via fallbacks; records without `overall` after all fallbacks are excluded from feature-engineered experiments.

## 6. Engineered Indicators for Bias & Purchasing Power
- Purpose: derive context-rich features that capture league/club purchasing power and groupings relevant to fairness analyses.
- Actions: compute league median transfer fee as a purchasing-power proxy, create age buckets, map position labels to position groups, and retain league and club-level summary indicators.
- Output: enriched feature set including `league_median_fee_to`, `age_bucket`, and `pos_group` to be used as model features or audit attributes.

## 7. Fairness Framework (Design and Metrics)
- Purpose: articulate the fairness evaluation objectives and define concrete metrics and groupings to audit model behavior.
- Key elements: define target, sensitive/context groups (e.g., `League_from`, `Nationality`, `Region`), minimum group sizes, and the metrics to compute (signed residuals, MAE, RMSE, calibration across predicted-fee bins).
- Output: a structured framework describing what constitutes potential bias and how to measure it programmatically.

## 8. Baseline Modeling (Linear Regression & Baseline Random Forest)
- Purpose: establish baseline predictive performance and error patterns for comparison with fairness-aware approaches.
- Actions: prepare modeling matrices (one-hot encoding), train a Linear Regression baseline and a baseline Random Forest (without engineered FIFA features), and record global MSE and R2.
- Output: baseline models and predictions used by downstream fairness/diagnostic cells.

## 9. Fairness-Aware Pre-Processing
- Purpose: create training/test splits and sample weights that reduce representation imbalance across sensitive groups while preserving predictive validity.
- Actions: keep context columns for audit but drop them from inputs; build a stratification key combining Region × target fee band; implement a robust fallback stratification chain (region×band → region only → unstratified); compute inverse-frequency sample weights by Region and fee band; normalize sample weights.
- Diagnostics: visualize train/test region representation, sample-weight distribution, and mean sample-weight per region.
- Output: `X_train_fair`, `X_test_fair`, `y_train_fair`, `y_test_fair`, `context_train`, `context_test`, and `sample_weight_fair`.

## 10. In-Processing Fairness Experiments (Model Variants)
- Purpose: compare model variants that incorporate fairness-aware choices during training (e.g., sample-weighted training) against standard models.
- Actions: train multiple Random Forest variants (standard, weighted, conservative), evaluate overall performance (RMSE, R2) and fairness disparities (signed residual gap and MAE gap across `Region`), and rank models by a combined fairness-performance criterion.
- Output: table of model comparisons and region-level fairness tables for each model.

## 11. Simpson's Paradox Diagnostics
- Purpose: detect and flag potential Simpson's paradox — cases where global group-level residual signs reverse within finer strata — which can mask important subgroup behavior.
- Actions: for the selected model, compute global residual summaries by Region, compute within-strata residuals (by League, by predicted-fee bin, and extended strata like AgeBucket and PositionBucket), compare sign consistency, and flag groups where sign consistency falls below a threshold.
- Visuals: heatmaps and bar charts that surface paradox rates across grouping pairs.
- Output: paradox diagnostic tables and visualizations guiding deeper investigation.

## 12. Hierarchical Strategy: Global Model + Group Residual Models
- Purpose: combine a robust global predictor with local residual-correction models for groups with adequate data, improving local accuracy without training entirely separate models.
- Actions: train a global Random Forest (using fairness-aware weights), compute residuals on the training set, and for Regions with sufficient samples fit lightweight residual models; generate hierarchical predictions on test data by adding region residual corrections when available.
- Decision logic: fallback to global-only predictions when region-specific residual models are not available.
- Output: hierarchical predictions and a comparison against single-model baselines on both performance and fairness metrics.

## 13. Explainability: SHAP, LIME, PDP, and ICE
- Purpose: provide global and local interpretability to support model debugging, transparency, and stakeholder communication.
- Actions: identify top features from the chosen model, sample a test subset for SHAP summary and dependence plots, run LIME on a hard-to-predict instance for local explanation, and produce PDP/ICE plots for selected continuous-like features to visualize marginal effects and heterogeneity.
- Output: SHAP beeswarm and bar charts, LIME explanation table for a selected case, PDP/ICE visualizations for chosen features.

## 14. Evaluation & Visualization Suite
- Purpose: provide a consistent set of visuals and tables for assessing both predictive quality and fairness across models.
- Typical artifacts: global performance metrics, signed residual tables by group, MAE/RMSE tables by group, calibration tables by predicted-fee quintile × region, residual distribution plots, fairness-performance frontier plots, and paradox heatmaps.

## 15. Artifacts, Reproducibility & Notes
- Artifacts: trained model objects, evaluation tables, fairness diagnostic tables, and plots are produced through the notebook; they should be exported (pickled or saved) if further programmatic use or deployment is intended.
- Reproducibility: ensure the `data/` zip files and a pinned dependency list (e.g., `requirements.txt`) are preserved; set random seeds for splits and model training as done in the notebook.
- Caveats: textual fuzzy matching and FIFA merges introduce potential false positives — review match rates and consider manual checks for high-value transfers.

## 16. Recommended Next Steps for Productionization
- Persist cleaned and feature-engineered datasets as versioned artifacts for reuse.
- Serialize the chosen model and any local residual models with metadata describing training date, sample sizes, and fairness metrics.
- Add unit tests around merging/fallback logic and a small integration test for the end-to-end pipeline using a deterministic sample.
- Build a scheduled audit job that re-runs the fairness diagnostics periodically and alerts on drifted metrics.

## References
- Source notebook: `notebooks/cleaned_model_pipeline.ipynb` (canonical reference for the implementation-level details).

---
Document created as a concise, non-code operational summary of the notebook workflow.
