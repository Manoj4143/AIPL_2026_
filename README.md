# AIPL 2026 - IPL Match Outcome Forecast

A live forecasting competition. You build a model on historical IPL ball-by-ball data, submit probabilistic predictions for upcoming IPL 2026 matches, and watch your rank shift live as each match plays out.

---

## The prediction task

For every match in the scoring set, predict the probability of each of **4 outcome classes**:

| Class | Definition |
|---|---|
| `A_small` | Team A wins by ≤ 20 runs **OR** by ≤ 5 wickets |
| `A_big` | Team A wins by > 20 runs **OR** by ≥ 6 wickets |
| `B_small` | Team B wins by ≤ 20 runs **OR** by ≤ 5 wickets |
| `B_big` | Team B wins by > 20 runs **OR** by ≥ 6 wickets |

The four probabilities must sum to 1.0 per match.

### Team A vs Team B convention — read carefully

The label of "Team A" and "Team B" is determined as follows:

- **For historical 2025 holdout matches (Public Leaderboard):** Team A = team batting first. Team B = team batting second. (Determined by the toss decision, which has already happened.)

- **For IPL 2026 scoring matches (Private Leaderboard):** Team A = home team (the team listed first in the official BCCI fixture). Team B = away team. **The toss has not happened at the time you submit predictions**, so Team A is *not* the team batting first for these matches — Team A is whoever the BCCI schedule calls the home team.

This asymmetry is deliberate. You're submitting predictions before the toss, so we anchor on the schedule-time team identity. Refer to `schedule.csv` for the canonical Team A / Team B assignment for the 5 scoring fixtures.

---

## Files in this bundle

```
train.zip
├── train_IPL.csv           Ball-by-ball data, 1,145 matches (2008 → mid-2025)
├── schedule.csv            The 5 IPL 2026 scoring fixtures
├── sample_submission.csv   53-row template with uniform 0.25 priors
├──public_lb_matches.csv   Pre-match metadata for the 48 held-out matches
└── README.md               This document
```

**Held-out matches (used for Public Leaderboard):**
- The 24 most recent 2025 matches (May 2 → June 3, 2025)
- The 24 played 2026 matches (March 28 → April 16, 2026)

These 48 matches are physically absent from `train_IPL.csv`. You don't see their ball-by-ball data, only their existence as match IDs in `sample_submission.csv`. 45 of them are scoreable (3 are abandoned/no-result and excluded from scoring).

---

## Data dictionary — `train_IPL.csv` (38 columns)

272,704 rows × 38 columns. One row per ball.

### Identifiers and timing

| Column | Description |
|---|---|
| `Match ID` | Unique integer identifier for the match |
| `Date` | Match date (YYYY-MM-DD) |
| `season` | IPL season (string, e.g. `"2024/25"`) |
| `Venue` | Stadium name (canonicalized — see note below) |
| `city` | City of the venue |

### Teams and toss

| Column | Description |
|---|---|
| `Bat First` | Team batting first this match (= Team A in historical labels) |
| `Bat Second` | Team batting second (= Team B in historical labels) |
| `toss_winner` | Team that won the toss |
| `toss_decision` | What the toss winner chose: `bat` or `field` |

### Ball-level events

| Column | Description |
|---|---|
| `Innings` | Innings number (1 = first innings, 2 = second innings) |
| `Over` | Over number (1-indexed) |
| `Ball` | Ball number within the over (1-indexed) |
| `Batter` | Player on strike for this ball |
| `Non Striker` | Player at non-striker's end |
| `Bowler` | Bowler delivering this ball |
| `Batter Runs` | Runs scored off the bat (excluding extras) |
| `Extra Runs` | Extras conceded on this ball (wides, no-balls, byes, leg-byes, penalty) |
| `Runs From Ball` | Total runs from this ball (`Batter Runs + Extra Runs`) |
| `Ball Rebowled` | 1 if the ball was a rebowl (e.g. wide), else 0 |
| `Extra Type` | List indicator for type of extra (e.g. `'wides'`, `'noballs'`, `'legbyes'`) |
| `Wicket` | 1 if a wicket fell on this ball, else 0 |
| `Dismissal Method` | How the batter was out (e.g. `caught`, `bowled`, `lbw`, `run out`) |
| `Player Out` | Player dismissed (NaN if no wicket) |
| `Valid Ball` | 1 if a legal delivery (counts toward the over), 0 if a wide/no-ball |

### Engineered cumulative columns (running state per innings)

These columns track innings state up to and including the current ball.

| Column | Description |
|---|---|
| `Innings Runs` | Cumulative runs in this innings up to this ball |
| `Innings Wickets` | Cumulative wickets fallen in this innings |
| `Target Score` | Target the chasing side needs to win (innings 1 final + 1) |
| `Runs to Get` | Runs the chasing side still needs (innings 2 only) |
| `Balls Remaining` | Legal balls remaining in this innings |
| `Total Batter Runs` | Cumulative runs scored by the batter on strike up to this ball |
| `Total Non Striker Runs` | Cumulative runs scored by the non-striker |
| `Batter Balls Faced` | Balls the on-strike batter has faced |
| `Non Striker Balls Faced` | Balls the non-striker has faced |
| `Player Out Runs` | Runs scored by the dismissed batter (NaN if no wicket) |
| `Player Out Balls Faced` | Balls faced by the dismissed batter |
| `Bowler Runs Conceded` | Cumulative runs conceded by this bowler in this innings |

### Match-level outcome columns

These are **constant across every ball of a single match** (`groupby('Match ID').first()` works cleanly).

| Column | Description |
|---|---|
| `match_won_by` | Name of the winning team (use this to derive Team A / Team B side of the label) |
| `result_type` | NaN for normal matches; `'tie'`, `'no result'` for edge cases |

> **Important:** `match_won_by` is part of the **target label**, not a feature. It is available for past matches so you can derive training labels. It will not be available at inference time for the held-out matches or the IPL 2026 fixtures.

---

## Deriving training labels — your first task

We have **intentionally not provided** a pre-computed 4-class label or a `win_outcome` margin string. You must derive both yourself from the data.

Match-level columns are constant per ball, so collapse with groupby:
Derive 4-class labels from columns in train_IPL.csv. You'll need to:
1. Collapse to a match-level view
2. Determine the winning side using match_won_by and Bat First
3. Determine the margin using cumulative innings totals (Innings Runs at the end of each innings, and Innings Wickets when the chase finishes)
4. Apply the 4-class rule (≤20 runs / ≤5 wickets = small)
Filter out matches with result_type = tie or no result.

After joining innings totals back into your match table, apply this function to get a 4-class label per match. Expect roughly 1,124 labelable matches out of 1,145 (the rest are ties, no-results, or abandoned).


### Edge cases worth knowing

- **Ties (`result_type='tie'`)**: 15 matches in the data. Often resolved by Super Over in real life, but here marked as ties. Exclude from training.
- **No-results (`result_type='no result'`)**: ~6 matches with no resolution. Exclude from training.
- **DLS-affected matches (Duckworth-Lewis applied)**: ~10-12 matches. The naive `inn1_runs - inn2_runs` margin computation will not match the official DLS-adjusted margin for these. We're not flagging them explicitly — handling them is up to you. Options: accept ~1% label noise, drop them via heuristic (e.g. innings 2 ended with > 5 balls remaining and innings 2 score < target), or use external sources to identify them.

---

## Submission format — `submission.csv`

Submit a CSV with exactly **53 rows × 5 columns**:

```
match_id,A_small,A_big,B_small,B_big
1473488,0.25,0.25,0.25,0.25
1473489,0.25,0.25,0.25,0.25
...
1527674,0.25,0.25,0.25,0.25
1527675,0.25,0.25,0.25,0.25
...
M_2026_T01,0.25,0.25,0.25,0.25
M_2026_T02,0.25,0.25,0.25,0.25
M_2026_T03,0.25,0.25,0.25,0.25
M_2026_T04,0.25,0.25,0.25,0.25
M_2026_T05,0.25,0.25,0.25,0.25
```

- Rows 1–24: held-out 2025 matches (numeric Match IDs)
- Rows 25–48: played 2026 matches (numeric Match IDs)
- Rows 49–53: IPL 2026 May scoring fixtures (`M_2026_T01` → `M_2026_T05`)
- Each row's 4 probability columns must sum to 1.0
- Probabilities must be in `[0, 1]`
- Match IDs must match exactly (same string format) as `sample_submission.csv`

A submission that's just `sample_submission.csv` (uniform 0.25 priors) scores `ln(4) ≈ 1.3863` — the trivial baseline.

---

## Evaluation metric — Mean Columnwise Log Loss

For each scored match, log loss is computed as:

```
loss_match = - sum over 4 classes c of: y_true[c] * log(y_pred[c])
```

Where `y_true` is the one-hot ground truth (1 in the correct class, 0 elsewhere) and `y_pred` is your predicted probability vector.

The competition score is the **mean** of `loss_match` across scored matches.

**Lower is better.**

Reference numbers:
- Uniform 0.25 prediction: `ln(4) ≈ 1.3863`
- Class-prior baseline (predict 2023+ historical class proportions for every match): around 1.32–1.35 on the 45-match Public LB
- A reasonably engineered ML model: 1.20–1.30 range
- Strong, well-calibrated submissions: 0.95–1.10 range

---

## Leaderboard mechanics

There are **two leaderboards**, both using the same metric.

### Public Leaderboard

Scored against **48 held-out matches** (45 of which are scoreable; 3 abandoned matches are excluded):
- **24 from the 2025 season** (May 2 → June 3, 2025) — 22 scoreable, 2 abandoned
- **24 from the 2026 season** (March 28 → April 16, 2026) — 23 scoreable, 1 abandoned

These rows are absent from `train_IPL.csv` — you cannot see their ball-by-ball data, only that the matches happened. Their outcomes are known to the host but never shared.

The Public LB is **visible during the build phase** and updates with every submission. It tells you roughly how your model is doing without you knowing the answers.

### Private Leaderboard

Scored against the **5 IPL 2026 scoring fixtures** (May 21–24). These matches haven't been played at submission lock time. Predictions are locked. As each match plays out in real life, the private leaderboard updates **live**.

The Private LB is **the actual evaluation**. Your final ranking is based on it.

### Submission limits

- **5 submissions per day** per team during the build phase
- **1 final submission** to be selected for Private LB scoring
- Submissions automatically lock at the build-phase deadline

---



## Note:

T20 cricket is genuinely high-variance. Even professional bookies' implied 4-class probabilities for IPL matches typically achieve log loss of 1.10–1.20. You're not expected to beat that, but you should be able to substantially beat the trivial baselines through thoughtful feature engineering and calibration.
