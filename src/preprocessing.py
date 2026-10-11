"""
Data Preprocessing and Feature Engineering Engine for IPL Match Outcome Prediction.

Preserves the existing ML preprocessing logic, label derivation,
team name normalization, and historical feature computation from AIPL.ipynb.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import catboost as cb

# Canonical entity consolidation mapping from AIPL.ipynb
TEAM_MAPPING = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

# The 4 preserved outcome classes
TARGET_CLASSES = ['A_small', 'A_big', 'B_small', 'B_big']

FEATURE_COLS = [
    'team_a_win_rate',
    'team_b_win_rate',
    'team_a_bat1st_rate',
    'venue_avg_score',
    'win_rate_diff',
    'Bat First_enc',
    'Bat Second_enc',
    'Venue_enc',
    'toss_decision_enc',
    'team_a_won_toss'
]


def clean_team_names(df, columns):
    """
    Standardize historical team name variants according to project convention.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(TEAM_MAPPING)
    return df_clean


def find_data_file(filename, search_dirs=None):
    """
    Locates a data file across standard locations (data/, project root, cwd).
    """
    if search_dirs is None:
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        project_dir = os.path.dirname(src_dir)
        search_dirs = [
            os.path.join(project_dir, 'data'),
            project_dir,
            os.path.join(os.getcwd(), 'data'),
            os.getcwd()
        ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_raw_ball_by_ball(filepath=None):
    """
    Loads and cleans raw ball-by-ball dataset.
    Raises FileNotFoundError with a clear message explaining that raw data is
    only required for retraining, while normal inference uses serialized model artifacts.
    """
    if filepath is None:
        filepath = find_data_file('train_IPL.csv')
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(
            "Raw training dataset 'train_IPL.csv' was not found in data/ or project root.\n\n"
            "[NOTE] Model Retraining vs Inference:\n"
            "- The raw ball-by-ball dataset ('train_IPL.csv') is strictly required ONLY for optional model retraining.\n"
            "- For normal match outcome inference and running the analytics dashboard, the pre-trained CatBoost model ('models/catboost_model.cbm') and serialized metadata are already provided and fully operational.\n"
            "- You do NOT need the raw training CSV to make predictions or run the web dashboard.\n\n"
            "Quick Run Options:\n"
            "  - Normal inference:\n"
            "      streamlit run app.py\n"
            "  - Run tests:\n"
            "      python -m unittest discover -s tests -p \"test_*.py\"\n"
            "  - Optional retraining (requires train_IPL.csv):\n"
            "      python -m src.preprocessing\n\n"
            "If you explicitly wish to retrain the model, please place 'train_IPL.csv' in the project root or 'data/' directory before running retraining."
        )

    df = pd.read_csv(filepath, low_memory=False)
    df = clean_team_names(df, ['Bat First', 'Bat Second', 'toss_winner', 'match_won_by'])
    return df


def aggregate_matches(bbb_df):
    """
    Aggregates ball-by-ball data to match-level DataFrame and innings statistics,
    exactly matching AIPL.ipynb Cell 15.
    """
    match_df = bbb_df.groupby('Match ID').agg({
        'Bat First': 'first',
        'Bat Second': 'first',
        'Venue': 'first',
        'city': 'first',
        'toss_winner': 'first',
        'toss_decision': 'first',
        'match_won_by': 'first'
    }).reset_index()

    innings_stats = bbb_df.groupby(['Match ID', 'Innings']).agg({
        'Runs From Ball': 'sum',
        'Extra Runs': 'sum',
        'Wicket': 'sum'
    }).reset_index()
    innings_stats['Total Runs'] = innings_stats['Runs From Ball'] + innings_stats['Extra Runs']

    return match_df, innings_stats


def derive_outcome_labels(match_df, innings_stats):
    """
    Derives 4-class outcome labels from match results and innings statistics:
      - A_big: Bat First team wins by > 20 runs
      - A_small: Bat First team wins by <= 20 runs
      - B_big: Bat Second team wins by >= 6 wickets
      - B_small: Bat Second team wins by <= 5 wickets
    """
    targets = []
    for _, row in match_df.iterrows():
        match_id = row['Match ID']
        team_a = row['Bat First']
        team_b = row['Bat Second']
        winner = row['match_won_by']

        match_innings = innings_stats[innings_stats['Match ID'] == match_id]
        try:
            runs_A = match_innings[match_innings['Innings'] == 1]['Total Runs'].values[0]
            wkts_B = match_innings[match_innings['Innings'] == 2]['Wicket'].values[0]
        except (IndexError, KeyError):
            targets.append(np.nan)
            continue

        if winner == team_a:
            inn2 = match_innings[match_innings['Innings'] == 2]
            runs_B = inn2['Total Runs'].values[0] if len(inn2) > 0 else 0
            margin_runs = runs_A - runs_B
            targets.append('A_big' if margin_runs > 20 else 'A_small')
        elif winner == team_b:
            margin_wickets = 10 - wkts_B
            targets.append('B_big' if margin_wickets >= 6 else 'B_small')
        else:
            targets.append(np.nan)

    df_labeled = match_df.copy()
    df_labeled['target'] = targets
    df_labeled = df_labeled.dropna(subset=['target']).reset_index(drop=True)
    return df_labeled


def compute_historical_profiles(match_df, innings_stats):
    """
    Computes historical statistics for teams and venues.
    """
    all_teams = [str(t) for t in pd.concat([match_df['Bat First'], match_df['Bat Second']]).unique() if pd.notna(t)]
    team_stats = {}
    for team in all_teams:
        total_matches = match_df[(match_df['Bat First'] == team) | (match_df['Bat Second'] == team)].shape[0]
        wins = match_df[match_df['match_won_by'] == team].shape[0]
        bat_1st_matches = match_df[match_df['Bat First'] == team]
        bat_1st_wins = bat_1st_matches[bat_1st_matches['match_won_by'] == team].shape[0]

        team_stats[team] = {
            'win_rate': float(wins / total_matches) if total_matches > 0 else 0.5,
            'bat_first_win_rate': float(bat_1st_wins / bat_1st_matches.shape[0]) if bat_1st_matches.shape[0] > 0 else 0.5,
            'total_matches': int(total_matches)
        }

    venue_avg_runs = innings_stats.groupby('Match ID')['Total Runs'].sum().reset_index().merge(
        match_df[['Match ID', 'Venue', 'city']], on='Match ID'
    )
    venue_profile = {
        str(k): float(v) for k, v in venue_avg_runs.groupby('Venue')['Total Runs'].mean().to_dict().items() if pd.notna(k)
    }

    # Venue to city mapping
    venue_to_city = {}
    for _, row in match_df.dropna(subset=['Venue']).iterrows():
        v = str(row['Venue'])
        c = row['city']
        if pd.notna(c) and v not in venue_to_city:
            venue_to_city[v] = str(c)

    return team_stats, venue_profile, venue_to_city


def fit_feature_encoders(match_df):
    """
    Fits label encoders for categorical features to guarantee reproducible encoding.
    """
    cat_encoders = {}
    cat_cols = ['Bat First', 'Bat Second', 'Venue', 'toss_decision']
    all_teams = sorted([str(t) for t in set(match_df['Bat First']).union(set(match_df['Bat Second'])) if pd.notna(t)])
    all_venues = sorted([str(v) for v in set(match_df['Venue']) if pd.notna(v)])
    all_decisions = ['bat', 'field']

    le_team = LabelEncoder()
    le_team.fit(all_teams)
    cat_encoders['Bat First'] = le_team
    cat_encoders['Bat Second'] = le_team

    le_venue = LabelEncoder()
    le_venue.fit(all_venues)
    cat_encoders['Venue'] = le_venue

    le_toss = LabelEncoder()
    le_toss.fit(all_decisions)
    cat_encoders['toss_decision'] = le_toss

    return cat_encoders


def transform_features(df, team_stats, venue_profile, cat_encoders):
    """
    Constructs feature matrix exactly adhering to project feature definitions.
    """
    df_feats = df.copy()

    df_feats['team_a_win_rate'] = df_feats['Bat First'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    df_feats['team_b_win_rate'] = df_feats['Bat Second'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    df_feats['team_a_bat1st_rate'] = df_feats['Bat First'].map(lambda x: team_stats.get(x, {}).get('bat_first_win_rate', 0.5))
    df_feats['venue_avg_score'] = df_feats['Venue'].map(lambda x: venue_profile.get(x, 330.0))
    df_feats['win_rate_diff'] = df_feats['team_a_win_rate'] - df_feats['team_b_win_rate']

    for col in ['Bat First', 'Bat Second', 'Venue', 'toss_decision']:
        le = cat_encoders[col]
        known_classes = set(le.classes_)
        df_feats[col + '_enc'] = df_feats[col].map(
            lambda x: le.transform([x])[0] if x in known_classes else -1
        )

    if 'toss_winner' in df_feats.columns:
        df_feats['team_a_won_toss'] = (df_feats['toss_winner'] == df_feats['Bat First']).astype(int)
    else:
        df_feats['team_a_won_toss'] = 0

    return df_feats[FEATURE_COLS]


def train_and_save_model(model_dir=None):
    """
    Executes full pipeline:
      1. Loads historical data
      2. Computes aggregations and labels
      3. Fits encoders and statistics
      4. Trains CatBoost model
      5. Serializes model to models/catboost_model.cbm and metadata to models/
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_dir is None:
        model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print("Loading ball-by-ball dataset...")
    bbb_df = load_raw_ball_by_ball()

    print("Aggregating matches and deriving outcome labels...")
    match_df, innings_stats = aggregate_matches(bbb_df)
    labeled_df = derive_outcome_labels(match_df, innings_stats)
    print(f"Total labeled matches: {len(labeled_df)}")

    print("Computing team and venue profiles...")
    team_stats, venue_profile, venue_to_city = compute_historical_profiles(match_df, innings_stats)
    cat_encoders = fit_feature_encoders(match_df)

    print("Transforming training features...")
    X = transform_features(labeled_df, team_stats, venue_profile, cat_encoders)

    le_target = LabelEncoder()
    # Ensure consistent order: ['A_big', 'A_small', 'B_big', 'B_small']
    le_target.fit(sorted(labeled_df['target'].unique()))
    y = le_target.transform(labeled_df['target'])

    print("Training CatBoost model...")
    model = cb.CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        loss_function='MultiClass',
        random_seed=42,
        verbose=0
    )
    model.fit(X, y)

    # Save CatBoost model
    model_path = os.path.join(model_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save metadata
    venue_matches = {str(k): int(v) for k, v in match_df['Venue'].value_counts().items()}
    metadata = {
        'feature_cols': FEATURE_COLS,
        'target_classes': list(le_target.classes_),
        'team_stats': team_stats,
        'venue_profile': venue_profile,
        'venue_to_city': venue_to_city,
        'all_teams': sorted(list(team_stats.keys())),
        'all_venues': sorted(list(venue_profile.keys())),
        'all_toss_decisions': ['bat', 'field'],
        'venue_matches': venue_matches
    }

    metadata_json_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    artifacts = {
        'metadata': metadata,
        'cat_encoders': cat_encoders,
        'le_target': le_target
    }
    joblib_path = os.path.join(model_dir, 'model_artifacts.joblib')
    joblib.dump(artifacts, joblib_path)
    print(f"Metadata and encoders saved to {joblib_path}")

    return model_path, joblib_path


if __name__ == '__main__':
    try:
        train_and_save_model()
    except FileNotFoundError as e:
        print("\n" + "=" * 76)
        print("[RETRAINING HALTED: RAW TRAINING DATASET MISSING]")
        print("=" * 76)
        print(str(e))
        print("=" * 76 + "\n")
        import sys
        sys.exit(1)
