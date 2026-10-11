# Project Decision Log: IPL Match Outcome Prediction System

This document logs architectural, algorithmic, and machine-learning decisions for the AIPL 2026 IPL Match Outcome Prediction project.

---

## Decision 1: CatBoost Model Preservation and Serialization
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The project README and `catboost_info` logs indicate CatBoost was selected as the primary ML model for multiclass prediction. However, no serialized `.cbm` model file was saved to disk in the repository, and the notebook experimentation had explored alternative frameworks.
- **Decision**: Train and serialize the CatBoost model using the existing match-level feature engineering pipeline and target labels (`iterations=200`, `learning_rate=0.05`, `depth=4`, `loss_function='MultiClass'`, `random_seed=42`). Export the model artifact to `models/catboost_model.cbm` along with metadata (`models/model_metadata.json` and `models/metadata.joblib`) containing team historical win-rates, venue scoring baselines, and categorical encodings.
- **Reason**: CatBoost natively handles multiclass probability predictions, aligns with the project documentation, reproduces the exact loss profile recorded in `catboost_info`, and provides fast, deterministic inference without needing to parse the raw 60MB ball-by-ball dataset on every request.
- **Alternatives Considered**:
  - Re-running Optuna hyperparameter searches across XGBoost/LightGBM ensembles (Rejected: Excessive overhead, does not adhere to CatBoost being the documented source of truth).
  - Loading raw CSVs and re-fitting on the fly inside Streamlit (Rejected: Slow, non-scalable, high latency).
- **Consequences**: Fast inference (<50ms), reliable serialization, zero dependency on training CSVs during production runtime.

---

## Decision 2: Team A / Team B Convention and Toss Dynamics Mapping
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The training dataset uses `Bat First` and `Bat Second` as the foundational entities, whereas user inputs and competition fixtures specify `team_a`, `team_b`, `toss_winner`, and `toss_decision`. The required target outputs are `A_small`, `A_big`, `B_small`, `B_big`.
- **Decision**:
  1. Determine batting order from user inputs:
     - If Toss Winner chooses to bat, `batting_first = toss_winner`, `batting_second = the other team`.
     - If Toss Winner chooses to field, `batting_first = the other team`, `batting_second = toss_winner`.
  2. Map to model features:
     - The ML engine evaluates `Bat First` vs `Bat Second`.
  3. Map probabilities back to `Team A` and `Team B`:
     - If `batting_first == team_a`: `A_small = P(Bat First wins small)`, `A_big = P(Bat First wins big)`, `B_small = P(Bat Second wins small)`, `B_big = P(Bat Second wins big)`.
     - If `batting_first == team_b`: `A_small = P(Bat Second wins small)`, `A_big = P(Bat Second wins big)`, `B_small = P(Bat First wins small)`, `B_big = P(Bat First wins big)`.
- **Reason**: This guarantees that `A_small` and `A_big` always represent Team A's win outcomes regardless of whether Team A bats first or chases, preserving semantic consistency for the user and alignment with competition rules.
- **Alternatives Considered**:
  - Forcing Team A to always bat first (Rejected: Unrealistic in real cricket, where toss determines innings).
  - Ignoring toss outcomes (Rejected: Degrades prediction accuracy and ignores available match context).
- **Consequences**: Consistent, correct team outcome mapping that accurately accounts for toss advantage.

---

## Decision 3: Handling Missing Features and Unseen Categories
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: If a user selects a newly introduced venue, an unseen city, or historical statistics are missing for a team/venue, the inference engine must handle this gracefully without crashing or silently producing nonsense.
- **Decision**:
  - Historical win rate default: 0.50 (neutral prior).
  - Bat-first win rate default: 0.50 (neutral prior).
  - Venue average score default: 330.0 runs (global historical aggregate across both innings).
  - Categorical encoders use explicit fallback encoding for unseen categories (`-1` or default index) and log a warning if encountered.
  - Optional `city` parameter is mapped to venue profiles when venue is known, and verified against historical records.
- **Reason**: Prevents application crashes while ensuring fallback values reflect neutral baseline priors.
- **Alternatives Considered**:
  - Imputing arbitrary 0 values (Rejected: Biases the tree splits strongly against unseen venues).
  - Rejecting unseen venues completely (Rejected: Limits user experimentation).
- **Consequences**: Robust inference across standard and edge-case inputs.

---

## Decision 4: Architectural Separation of Concerns
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Phase 1 requires an interactive Streamlit UI while maintaining modular, reusable, testable code.
- **Decision**:
  - `src/preprocessing.py`: Core data aggregation, team cleaning, label generation, feature engineering, and training pipeline.
  - `src/prediction.py`: Standalone inference service exposing `predict_match(...)`, input validation, model loading, and probability checks.
  - `app.py`: Pure presentation layer for the Streamlit dashboard, consuming `predict_match(...)` from `src/prediction.py`.
- **Reason**: Adheres to clean code principles, enables automated unit testing without Streamlit runtime dependencies, and sets up a clean foundation for Phase 2.
- **Alternatives Considered**:
  - Putting all inference and data preparation directly in `app.py` (Rejected: Poor maintainability and testability).
- **Consequences**: Highly testable, modular architecture.

---

## Decision 5: Phase 1 Milestone Completion and Sign-Off
- **Date**: 2026-10-11
- **Status**: Completed / Accepted
- **Context**: Verification and completion of Phase 1 requirements: local interactive Streamlit web application (`app.py`), modular prediction service (`src/prediction.py`), preprocessing and model serialization (`src/preprocessing.py`), CatBoost model artifact (`models/catboost_model.cbm`), unit test suite (`tests/test_phase1.py`), and documentation updates.
- **Decision**: Sign off on Phase 1 vertical slice after passing automated test suite (7/7 tests passing) and browser subagent validation of user workflows (input validation, same-team prevention, probability distribution rendering, sum-to-100% check, and variance disclaimer).
- **Reason**: All Phase 1 deliverables meet specifications without modifying the source-of-truth CatBoost engine or label definitions.
- **Alternatives Considered**: None.
- **Consequences**: System is ready for local deployment and Phase 2 enhancements (form factors, advanced team-level metrics, and calibration tuning).

---

## Decision 6: Native CatBoost TreeSHAP Explainability and Feature Summary Panel
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Phase 2 requires explaining model predictions to show why the model favors Team A or Team B, while avoiding unstable or heavy external dependencies.
- **Decision**: Utilize CatBoost's native C++ implementation of TreeSHAP via `model.get_feature_importance(cb.Pool(X), type='ShapValues')` combined with a transparent feature-summary panel (historical win rates, relative strength differential, venue average runs, toss choice, and batting order).
- **Reason**: CatBoost natively supports exact TreeSHAP attribution for multiclass models without requiring the external `shap` package, eliminating heavy dependencies, preventing C++ compilation mismatch issues, and executing in <10ms.
- **Alternatives Considered**:
  - Installing the full Python `shap` package (Rejected: Unnecessary 150MB+ dependency and potential runtime instability; CatBoost's native TreeSHAP is faster and self-contained).
  - Simple static feature importance only (Rejected: Does not explain local single-prediction attributions).
- **Consequences**: Exact, per-prediction feature attribution charts and transparent input summaries without extra dependencies.

---

## Decision 7: Venue Metadata Enrichment with Genuine Match Counts
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Users need venue context (historical average runs, host city, training match sample size, and high-scoring indicator) without fabricated statistics.
- **Decision**: Calculate genuine match counts for all 39 venues from `train_IPL.csv` and store them in `models/model_metadata.json` alongside `venue_profile` and `venue_to_city`. Designate venues with average scores > 330.0 runs as "High-Scoring" and ≤ 330.0 runs as "Balanced / Bowler-Friendly".
- **Reason**: Provides accurate, grounded venue metrics without altering the trained CatBoost model weights or feature inputs.
- **Alternatives Considered**:
  - Re-querying the 60MB raw CSV file inside Streamlit on every venue selection (Rejected: Adds 200-500ms latency to UI dropdowns).
- **Consequences**: Instant, accurate venue stats displayed in the dashboard without runtime overhead.

---

## Decision 8: Session-State Prediction History Management
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: Users want to track multiple scenarios and compare match predictions within their active browser session.
- **Decision**: Store prediction records in `st.session_state.prediction_history` as an in-memory list of dictionaries, displaying a clear tabular summary and a "Clear History" button.
- **Reason**: Adheres to Phase 2 constraints against writing user predictions to disk or external services while providing a responsive audit log during experimentation.
- **Alternatives Considered**:
  - Appending predictions to a local CSV file (Rejected: Violates the Phase 2 specification restricting history to session-level state).
- **Consequences**: Clean, non-persistent, privacy-preserving session history.

---

## Decision 9: Multi-Tab Layout Architecture for Phase 2 Analytics Dashboard
- **Date**: 2026-10-11
- **Status**: Accepted
- **Context**: The application is expanding from a simple prediction form into an analytics dashboard featuring predictions, model explanations, team head-to-head comparisons, venue insights, and session history.
- **Decision**: Structure `app.py` into 5 cohesive tabs:
  1. `🔮 Match Prediction`: Input controls, outcome cards, probability charts, and tables.
  2. `🔍 Model Explanation`: Transparent feature inputs, batting order dynamics, and native SHAP attribution chart.
  3. `⚔️ Team Comparison`: Side-by-side win rate metrics and relative strength breakdown.
  4. `🏟️ Venue Insights`: Venue metrics (average score, city, historical matches, pace/scoring profile).
  5. `📜 Session History`: Record table with timestamp, match configuration, and Clear History trigger.
- **Reason**: Keeps the interface organized, uncluttered, professional, and accessible.
- **Alternatives Considered**:
  - A long single-page scrolling layout (Rejected: Poor usability and excessive scrolling).
- **Consequences**: Clean, modern, highly interactive UI that looks like a production-grade analytics product.

---

## Decision 10: Phase 2 Milestone Completion and Sign-Off
- **Date**: 2026-10-11
- **Status**: Completed / Accepted
- **Context**: Verification and completion of Phase 2 requirements: transformed Streamlit application into an analytics dashboard across 5 tabs (Match Prediction, Model Explanation, Team Comparison, Venue Insights, Session History), native CatBoost TreeSHAP feature attributions, head-to-head win rate profiles, genuine venue match counts and scoring category flags, in-memory session prediction history tracking and clearing, extended automated test suite (17/17 tests passing), and documentation updates.
- **Decision**: Sign off on Phase 2 deliverables after verifying end-to-end functionality, model preservation (weights and inference untouched), strict probability summation to 100%, and robust fallback handling for missing metadata.
- **Reason**: All Phase 2 functional and architectural requirements are met without introducing unstable dependencies or violating model integrity rules.
- **Alternatives Considered**: None.
- **Consequences**: The system is fully polished and ready for Phase 3 deployment packaging (Docker, FastAPI endpoints, and cloud hosting).


