# IPL Match Outcome Prediction System

A probabilistic machine-learning system that predicts IPL match outcomes using historical ball-by-ball cricket data.

The system predicts not only which team is likely to win, but also whether the victory margin will be small or large.

## Project Overview

This project uses historical IPL match data from 2008 onwards to train a multiclass classification model.

For every match, the model predicts probabilities for four possible outcomes:

| Class | Meaning |
|---|---|
| `A_small` | Team A wins by a small margin |
| `A_big` | Team A wins by a large margin |
| `B_small` | Team B wins by a small margin |
| `B_big` | Team B wins by a large margin |

A small win is defined as:

- Winning by 20 runs or fewer, or
- Winning by 5 wickets or fewer

A big win is defined as:

- Winning by more than 20 runs, or
- Winning by 6 or more wickets

The prediction probabilities for every match sum to 1.0.

## Objectives

The main objectives of this project are:

1. Transform ball-by-ball IPL data into match-level features.
2. Derive meaningful outcome labels from historical match results.
3. Train a multiclass machine-learning model.
4. Generate calibrated outcome probabilities.
5. Evaluate the model using multiclass log loss.
6. Provide an interactive web interface for match predictions.
7. Deploy the application so that users can test predictions easily.

## Machine-Learning Workflow

```text
Historical ball-by-ball data
            ↓
Data cleaning and preprocessing
            ↓
Match-level feature engineering
            ↓
Outcome-label generation
            ↓
Model training
            ↓
Probability prediction
            ↓
Model evaluation
            ↓
Interactive web application
```

## Dataset

The primary dataset contains historical IPL ball-by-ball information.

Important fields include:

- Match ID
- Match date
- Season
- Venue
- City
- Batting-first team
- Batting-second team
- Toss winner
- Toss decision
- Batter
- Bowler
- Runs scored
- Extras
- Wickets
- Overs
- Innings totals
- Balls remaining
- Match result

The dataset contains information from multiple IPL seasons and allows the model to learn patterns related to team performance, venue, toss decisions, scoring, and match margins.

## Feature Engineering

The ball-level data is aggregated into match-level information.

The feature-engineering process includes:

- Team identities
- Venue and city
- Toss winner
- Toss decision
- First-innings score
- Second-innings score
- Total wickets
- Run rate
- Runs required
- Balls remaining
- Wickets lost
- Historical team performance
- Historical venue performance
- Batting-first and chasing patterns

The target variable is generated from the winning team and the winning margin.

Matches with ties, abandoned results, or no-results are excluded from supervised training where an outcome label cannot be determined reliably.

## Model

The main model used in this project is CatBoost.

CatBoost was selected because:

- It works well with categorical variables.
- It can model nonlinear relationships.
- It requires less manual encoding for categorical features.
- It performs well on structured tabular data.
- It supports multiclass probability prediction.

The model performs four-class classification and returns a probability distribution for every match.

Example prediction:

```text
A_small: 0.34
A_big:   0.18
B_small: 0.29
B_big:   0.19
```

The highest probability represents the model's most likely outcome.

## Evaluation Metric

The main evaluation metric is multiclass log loss.

Log loss evaluates the quality of predicted probabilities rather than only the final predicted class.

Lower log loss is better.

The expected baseline values are:

| Model | Expected Log Loss |
|---|---:|
| Uniform random prediction | Approximately 1.3863 |
| Class-prior baseline | Approximately 1.32–1.35 |
| Reasonably engineered model | Approximately 1.20–1.30 |
| Strong calibrated model | Approximately 0.95–1.10 |

The project compares the CatBoost model against simple baselines to verify that the machine-learning model provides meaningful improvement.

## Model Validation

The validation process includes:

- Time-aware train-validation splitting
- Comparison against a uniform baseline
- Comparison against a class-prior baseline
- Multiclass log-loss evaluation
- Prediction probability validation
- Confusion matrix analysis
- Feature-importance analysis
- Class-distribution analysis

The final prediction probabilities are checked to ensure:

```text
0 ≤ probability ≤ 1
sum of four probabilities = 1
```

## Interactive Web Application

The planned interactive web application will allow users to enter match details and receive real-time predictions.

### User Inputs

The application will support inputs such as:

- Team A
- Team B
- Venue
- City
- Toss winner
- Toss decision
- Optional match conditions

### Application Output

The application will display:

- Predicted most likely outcome
- Probability of each outcome
- Probability chart
- Team comparison
- Model explanation
- Prediction confidence
- Important features influencing the prediction

Example:

```text
Most likely outcome: Team A wins by a small margin

Team A small win: 34%
Team A big win:   18%
Team B small win: 29%
Team B big win:   19%
```

### Planned Application Architecture

```text
User input
    ↓
Input validation
    ↓
Feature transformation
    ↓
Trained CatBoost model
    ↓
Probability prediction
    ↓
Charts and prediction explanation
```

### Recommended Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Streamlit
- Matplotlib
- Seaborn
- Plotly

Streamlit is suitable because it allows the trained model to be converted into a web application quickly without requiring a separate frontend framework.

## Suggested Application Structure

```text
AIPL_2026_/
├── app.py
├── AIPL.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── train_IPL.csv
│   └── schedule.csv
├── models/
│   └── catboost_model.cbm
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── prediction.py
├── submissions/
│   └── submission_final.csv
└── assets/
    └── screenshots/
```

## Example Streamlit Usage

After installing the dependencies, run:

```bash
streamlit run app.py
```

The application will open in a browser and provide an interactive prediction interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/Manoj4143/AIPL_2026_.git
cd AIPL_2026_
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook AIPL.ipynb
```

Run the interactive application:

```bash
streamlit run app.py
```

## Requirements

The project requires:

```text
pandas
numpy
scikit-learn
catboost
matplotlib
seaborn
plotly
streamlit
jupyter
```

## Submission Format

The prediction output follows the required format:

```text
match_id,A_small,A_big,B_small,B_big
```

Each row contains:

- Match ID
- Probability of Team A winning by a small margin
- Probability of Team A winning by a large margin
- Probability of Team B winning by a small margin
- Probability of Team B winning by a large margin

The four probability columns must sum to 1.0 for every match.

## Example Prediction Validation

```python
probability_columns = [
    "A_small",
    "A_big",
    "B_small",
    "B_big"
]

assert predictions[probability_columns].between(0, 1).all().all()

assert np.allclose(
    predictions[probability_columns].sum(axis=1),
    1.0,
    atol=1e-6
)
```

## Current Project Status: Phase 2 Completed

### Completed

- **Interactive IPL Analytics Dashboard (`app.py`)**:
  - Structured 5-tab user interface:
    1. `🔮 Match Prediction`: Input parameters, highlight outcome card, probability metrics, interactive Plotly bar chart, breakdown table, and 100% sum verification.
    2. `🔍 Model Explanation`: Transparent feature inputs, batting order dynamics, and native CatBoost TreeSHAP attribution chart.
    3. `⚔️ Team Comparison`: Side-by-side win rate metrics, bat-first win rates, total matches, and strength differential bar chart.
    4. `🏟️ Venue Insights`: Venue scoring profile, host city, training match sample size, and high-scoring ground classification.
    5. `📜 Session History`: Audit log of user predictions within the active session with a clear-history trigger.
- **Explainability Engine (`src/prediction.py`)**:
  - Integrated native CatBoost C++ TreeSHAP attribution (`type='ShapValues'`) without requiring heavy or unstable external packages.
  - Per-prediction feature contribution analysis showing which factors pushed probabilities higher or lower.
- **Historical Team & Venue Baselines**:
  - Team win rates and bat-first win rates computed from historical match records.
  - Genuine match sample sizes for all 39 venues.
- **Robust Input Validation & Error Boundaries**:
  - Prevents selecting identical teams for Team A and Team B.
  - Automatic toss-order and innings sequence resolution.
  - Fallback encoding for unseen categories and missing optional metadata.
- **Automated Test Suite (`tests/`)**:
  - 17 unit tests passing in `tests/test_phase1.py` and `tests/test_phase2.py`.
  - Tests covering predictions, toss-order dynamics, TreeSHAP generation, team comparisons, venue metrics, and history management.
- **Decision Tracking (`decision.md`)**:
  - Complete decision records from Decision 1 through Decision 10.

### In Progress

- Phase 3 deployment packaging preparation.

### Planned (Phase 3 & Deployment)

- Docker containerization for local and cloud deployment.
- High-performance REST API endpoint using FastAPI.
- Live odds or match monitoring integrations.
- Player-level Playing XI statistics and recent-form momentum indices.

## Dashboard Demonstration & Screenshots

```text
+-----------------------------------------------------------------------------------+
|  🏏 IPL Match Outcome Prediction System                                            |
|  Probabilistic ML Analytics Dashboard powered by CatBoost                          |
+-----------------------------------------------------------------------------------+
|  [🔮 Match Prediction] [🔍 Model Explanation] [⚔️ Team Comparison] [🏟️ Venue] ... |
+-----------------------------------------------------------------------------------+
|  Match Configuration:                                                             |
|  Team A: Chennai Super Kings           Team B: Gujarat Titans                     |
|  Venue:  MA Chidambaram Stadium        Toss:   CSK (bat)                          |
|                                                                                   |
|  [ 🔮 Predict Match Outcome ]                                                     |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |  MOST LIKELY OUTCOME: B_big (31.91%)                                        |  |
|  |  Gujarat Titans wins by a large margin (> 20 runs or >= 6 wickets)          |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
|  Probabilities:  A_small: 26.01% | A_big: 22.29% | B_small: 19.80% | B_big: 31.91%|
|  [====================== Probability Distribution Chart ======================]   |
|  ✅ Probability Integrity Validated: Probabilities sum to 100% (1.000000).        |
+-----------------------------------------------------------------------------------+
```

*(Screenshots can be stored in `assets/screenshots/`)*

## Running the Application

### Execution Commands

```text
Normal inference:
streamlit run app.py

Run tests:
python -m unittest discover -s tests -p "test_*.py"

Optional retraining:
python -m src.preprocessing
```

> **Important Note on Inference vs. Retraining:**
> - **Normal Inference & Web App**: Running `streamlit run app.py` and running the test suite (`python -m unittest discover -s tests -p "test_*.py"`) relies exclusively on the pre-trained serialized model artifacts (`models/catboost_model.cbm`, `models/model_artifacts.joblib`, and `models/model_metadata.json`). The raw training dataset is **not** required for making predictions.
> - **Optional Retraining**: The command `python -m src.preprocessing` is strictly optional and only intended for training a new model from scratch. It requires the raw ball-by-ball dataset (`train_IPL.csv`) to be placed in `data/` or the project root. If `train_IPL.csv` is absent, the script outputs a clear informational message recommending normal inference using the existing model artifacts.

## Limitations

Cricket is a high-variance sport, and match predictions are inherently uncertain.

The current system has several limitations:

- Player availability may not be known in advance.
- Playing XI information may be unavailable.
- Injuries and substitutions are difficult to model.
- Team strength changes between seasons.
- Weather and pitch conditions may not be fully represented.
- Toss information may not be available before the match.
- DLS-affected matches can introduce label noise.
- Historical patterns may not always apply to future seasons.

## Future Improvements

Potential improvements include:

- Recent team-form features
- Player-level batting and bowling statistics
- Expected playing XI information
- Player injury and availability data
- Venue-specific performance
- Weather and pitch information
- Elo-style team ratings
- Probability calibration
- Ensemble models
- Automated model retraining
- Prediction monitoring after every match
- REST API deployment
- Cloud deployment

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for sports prediction.

It combines:

- Real-world structured data
- Feature engineering
- Multiclass classification
- Probabilistic prediction
- Model evaluation
- Interactive deployment

The final goal is to provide a practical prediction system that allows users to enter match information and understand the probability of each possible match outcome.

## Author

Developed by Manoj4143 as an IPL machine-learning prediction project.