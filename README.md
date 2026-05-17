# 🏏 AIPL 2026 - IPL Match Outcome Forecast

> **Predict IPL 2026 match outcomes with machine learning** | Live forecasting competition scoring in real-time as matches play out

A data-driven approach to forecasting Indian Premier League (IPL) 2026 matches using **1,145 historical ball-by-ball records**, advanced feature engineering, and XGBoost probabilistic modeling. Submit predictions before the toss; watch your rank update live as each match plays out.

---

## 📊 The Challenge

For every match, predict **4 probability outcomes** that must sum to **1.0**:

| Outcome | When? |
|---------|-------|
| **A_small** 🟢 | Team A wins by ≤20 runs OR ≤5 wickets |
| **A_big** 🟢 | Team A wins by >20 runs OR ≥6 wickets |
| **B_small** 🔴 | Team B wins by ≤20 runs OR ≤5 wickets |
| **B_big** 🔴 | Team B wins by >20 runs OR ≥6 wickets |

### Key Point: Team A vs Team B

- **Public Leaderboard (2025 holdout matches):** Team A = team batting first
- **Private Leaderboard (IPL 2026 fixtures):** Team A = home team (from BCCI schedule)
- *Toss hasn't happened yet for 2026 matches — predictions anchor on schedule identity, not batting order*

---

## 📁 Project Files & Structure

| File | Description | Rows |
|------|-------------|------|
| **`AIPL.ipynb`** | Main Jupyter notebook with XGBoost model, feature engineering, hyperparameter tuning (Optuna), and cross-validation | Executable |
| **`train_IPL.csv`** | **272,704** ball-by-ball records from **1,145 IPL matches** (2008–2025) — used for training | 272.7K |
| **`schedule.csv`** | **5 upcoming IPL 2026 matches** to predict (May 21–24, 2026) | 5 |
| **`public_lb_matches.csv`** | Metadata for **48 held-out matches** used to validate on Public Leaderboard | 48 |
| **`sample_submission.csv`** | Template submission with uniform **0.25 priors** for all classes (baseline) | 53 |
| **`submission.csv`** | Your trained model's **53 predictions** (blend of public + private fixtures) | 53 |
| **`submission_fixed.csv`** | Refined submission after validation & calibration tuning | 53 |

### Dataset Breakdown

- **Training Data:** 1,145 matches → ~1,124 after removing ties/no-results
- **Public Leaderboard:** 48 matches (24 from 2025 playoffs, 24 from early 2026)
- **Private Leaderboard:** 5 upcoming IPL 2026 fixtures
- **Total Predictions:** 53 match outcomes

---

## 🚀 What This Model Does

### Key Components (from `AIPL.ipynb`):

1. **Data Cleaning & Entity Consolidation**
   - Standardizes team names (e.g., `Royal Challengers Bangalore` → `Royal Challengers Bengaluru`)
   - Handles name changes across IPL seasons

2. **Target Derivation**
   - Aggregates ball-by-ball data to match level
   - Extracts innings totals & wickets
   - Derives 4-class targets based on win margin rules
   - Filters out ties, no-results, abandoned matches

3. **Feature Engineering**
   - Toss outcomes
   - Venue characteristics
   - Historical team performance
   - Batter & bowler statistics
   - Aggregated match-level summaries

4. **XGBoost Classification**
   - **Classifier:** Multi-class probabilistic predictions
   - **Hyperparameter Tuning:** Optuna for optimal learning rate, depth, regularization
   - **Cross-Validation:** StratifiedKFold to ensure balanced class representation
   - **Output:** Calibrated probability distributions per class

5. **Evaluation**
   - **Metric:** Mean Columnwise Log Loss
   - **Formula:** $-\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{4} y_i^c \log(\hat{p}_i^c)$
   - **Lower is better**

---

## 📈 Performance Baseline

| Approach | Log Loss Score |
|----------|----------------|
| 🟡 Uniform 0.25 priors | ~**1.386** (trivial baseline) |
| 🟡 Historical class proportions | ~**1.32–1.35** |
| 🟢 Reasonably tuned ML model | ~**1.20–1.30** |
| 🟢 Strong, well-calibrated model | **0.95–1.10** |
| 📊 Professional bookies | ~**1.10–1.20** |

*T20 is inherently high-variance. Goal: beat trivial baselines through feature engineering and probability calibration.*

---

## 🎯 Leaderboards

### Public Leaderboard ✅ Visible During Build
- **48 held-out matches** (45 scoreable, 3 abandoned)
  - 24 from 2025 playoffs (May 2 – June 3)
  - 24 from early 2026 (March 28 – April 16)
- Updates with every submission
- Shows live performance without revealing answers
- Helps tune features & calibration

### Private Leaderboard 🔴 The Final Judge
- **5 IPL 2026 fixtures** (May 21–24)
- Locked at submission deadline
- Scores **live** as matches play out in real time
- **Final ranking based on Private LB only**

---

## 📝 Submission Format

Create a CSV with **exactly 53 rows × 5 columns**:

```csv
match_id,A_small,A_big,B_small,B_big
1473488,0.19801,0.26152,0.20142,0.33905
1473489,0.19345,0.24781,0.19660,0.36214
...
M_2026_T01,0.25,0.25,0.25,0.25
M_2026_T02,0.25,0.25,0.25,0.25
...
M_2026_T05,0.25,0.25,0.25,0.25
```

**Rules:**
- Rows 1–24: 2025 holdout matches (numeric IDs)
- Rows 25–48: 2026 early matches (numeric IDs)  
- Rows 49–53: IPL 2026 scoring fixtures (`M_2026_T01` → `M_2026_T05`)
- **Each row must sum to 1.0**
- **Probabilities in [0, 1]**
- **Match IDs must match exactly** (case-sensitive)

---

## ⚙️ Running the Pipeline

1. **Open `AIPL.ipynb`** in Jupyter/VS Code
2. **Execute cells sequentially:**
   - Load libraries & data
   - Clean team names
   - Aggregate to match level & derive targets
   - Engineer features
   - Train XGBoost with Optuna hyperparameter tuning
   - Generate probabilistic predictions
3. **Export predictions** → `submission.csv`
4. **Submit** (5 submissions per day limit during build phase)

---

## 🔍 Data Dictionary (Ball-by-Ball Records)

**38 columns per row; one row = one ball delivered**

### Core Identifiers
- `Match ID` — Unique match identifier
- `Date`, `season` — When & in which IPL season
- `Venue`, `city` — Stadium information

### Teams & Toss
- `Bat First`, `Bat Second` — Teams in match order
- `toss_winner`, `toss_decision` — Who won toss & chose to bat/field

### Ball-Level Events
- `Innings` — 1st or 2nd innings
- `Over`, `Ball` — Over (1–20) and ball (1–6) number
- `Batter`, `Non Striker`, `Bowler` — Players involved
- `Batter Runs`, `Extra Runs` — Runs breakdown
- `Wicket`, `Dismissal Method`, `Player Out` — Dismissal info

### Engineered Cumulative Features
- `Innings Runs`, `Innings Wickets` — Current innings state
- `Target Score`, `Runs to Get` — Chase dynamics
- `Balls Remaining` — Overs left
- `Total Batter Runs`, `Batter Balls Faced` — Batter performance to date

### Match Outcomes (Constant per Match)
- `match_won_by` — Winning team name
- `result_type` — "tie", "no result", or null (normal match)

> **Critical:** `match_won_by` is **target information** — available in training data to derive labels, but not at inference time.

---

## ⚠️ Edge Cases & Data Quality

| Issue | Count | Action |
|-------|-------|--------|
| **Ties** (`result_type='tie'`) | ~15 | Exclude from training |
| **No-results** (`result_type='no result'`) | ~6 | Exclude from training |
| **DLS-affected matches** | ~10–12 | Naive margin calculation will have ~1% noise; consider dropping or external validation |

---

## 🎓 Key Insights

- **Feature engineering > raw model complexity** — Well-engineered toss, venue, & historical stats beat generic XGBoost
- **Probability calibration is critical** — Raw model outputs often need temperature scaling or isotonic regression
- **Class imbalance matters** — Stratified K-fold cross-validation helps avoid skewed validation splits
- **Venue matters in T20** — Short grounds & ground conditions significantly impact margin distributions

---

## 📞 Competition Rules Summary

- ✅ **Public Leaderboard** updates live with every submission (visible anytime)
- ✅ **5 submissions per day** during build phase
- ✅ **1 final submission** locked for Private LB scoring
- ✅ **Private LB scores live** as real IPL 2026 matches play out (May 21–24)
- ❌ No real-time ball-by-ball data for 2026 fixtures before toss
- ❌ Predictions lock at deadline — **no mid-match updates allowed**

---

**Build your edge through meticulous feature engineering and probability calibration. The leaderboard updates live. May your predictions be well-calibrated! 🏆**
