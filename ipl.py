# ==============================================================================
# Production IPL Pipeline: Breaking the 1.0 Log Loss Barrier
# ==============================================================================

import sys
import subprocess
import warnings
from pathlib import Path
import os

# --- 1. Auto-Dependency Installer ---
def install_dependencies():
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost']
    print("Checking system dependencies...")
    for package in required_packages:
        try:
            import_name = 'sklearn' if package == 'scikit-learn' else package
            __import__(import_name)
        except ImportError:
            print(f"Installing missing package: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

install_dependencies()

# --- 2. Imports ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

# --- 3. Configuration ---
PROB_COLS = ['A_small', 'A_big', 'B_small', 'B_big']

# --- 4. Robust Probability Normalization ---
def normalize_probabilities(preds_df):
    """Strictly forces rows to sum to exactly 1.0 to prevent Kaggle penalties."""
    preds_df['A_small'] = preds_df['A_small'].round(4)
    preds_df['A_big'] = preds_df['A_big'].round(4)
    preds_df['B_small'] = preds_df['B_small'].round(4)
    
    preds_df['B_big'] = (1.0 - preds_df[['A_small', 'A_big', 'B_small']].sum(axis=1)).round(4)
    preds_df['B_big'] = preds_df['B_big'].clip(lower=0.0)
    return preds_df

# --- 5. Real Feature Engineering Engine ---
def build_cricket_features(train_df, test_meta):
    """
    Computes real, toss-agnostic historical features for teams and venues.
    """
    print("Calculating historical team and venue statistics...")
    
    # Standardize column naming variations across datasets
    for df in [train_df, test_meta]:
        df.columns = [c.lower().strip() for c in df.columns]
    
    # Identify critical columns dynamically
    team_a_col = 'team_a' if 'team_a' in train_df.columns else ('team1' if 'team1' in train_df.columns else None)
    team_b_col = 'team_b' if 'team_b' in train_df.columns else ('team2' if 'team2' in train_df.columns else None)
    venue_col = 'venue' if 'venue' in train_df.columns else None
    target_col = 'outcome_class' if 'outcome_class' in train_df.columns else None
    
    if not team_a_col or not team_b_col:
        raise ValueError("Could not find Team columns. Ensure columns are named 'Team_A' and 'Team_B'.")

    # 1. Calculate Baseline Global Team Win Rates from training data
    # Pre-calculate what classes mean general wins
    train_df['a_won'] = train_df[target_col].astype(str).str.startswith('A').astype(int)
    train_df['b_won'] = train_df[target_col].astype(str).str.startswith('B').astype(int)
    
    team_stats = {}
    all_teams = set(train_df[team_a_col]).union(set(train_df[team_b_col]))
    
    for team in all_teams:
        a_matches = train_df[train_df[team_a_col] == team]
        b_matches = train_df[train_df[team_b_col] == team]
        
        total_wins = a_matches['a_won'].sum() + b_matches['b_won'].sum()
        total_matches = len(a_matches) + len(b_matches)
        
        team_stats[team] = total_wins / total_matches if total_matches > 0 else 0.5

    # 2. Map calculated metrics back onto Train and Test datasets
    def apply_metrics(target_df):
        features_df = pd.DataFrame(index=target_df.index)
        
        # Win Rates
        features_df['team_a_global_winrate'] = target_df[team_a_col].map(team_stats).fillna(0.5)
        features_df['team_b_global_winrate'] = target_df[team_b_col].map(team_stats).fillna(0.5)
        features_df['winrate_diff'] = features_df['team_a_global_winrate'] - features_df['team_b_global_winrate']
        
        # Categorical structural codes for tree-based splits
        features_df['team_a_encoded'] = pd.factorize(target_df[team_a_col])[0]
        features_df['team_b_encoded'] = pd.factorize(target_df[team_b_col])[0]
        if venue_col:
            features_df['venue_encoded'] = pd.factorize(target_df[venue_col])[0]
        else:
            features_df['venue_encoded'] = 0
            
        return features_df

    X_train = apply_metrics(train_df)
    X_test = apply_metrics(test_meta)
    
    # Extract target cleanly
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col].astype(str))
    
    return X_train, y_train, X_test, le

# --- 6. Main Execution Pipeline ---
def run_pipeline():
    print("Initializing Match-Level Predictive Pipeline...")
    
    try:
        # 1. Check for files
        if not os.path.exists('sample_submission.csv'):
            raise FileNotFoundError("Missing 'sample_submission.csv'")
        if not os.path.exists('train_IPL.csv'):
            raise FileNotFoundError("Missing 'train_IPL.csv'")
            
        # Find match metadata for test set tracking
        if os.path.exists('schedule.csv'):
            meta_file = 'schedule.csv'
        elif os.path.exists('public_lb_matches.csv'):
            meta_file = 'public_lb_matches.csv'
        else:
            raise FileNotFoundError("Could not find 'schedule.csv' or 'public_lb_matches.csv' to identify test teams.")

        # 2. Read datasets
        sample_sub = pd.read_csv('sample_submission.csv')
        train_df = pd.read_csv('train_IPL.csv')
        test_meta_all = pd.read_csv(meta_file)
        
        # 3. Filter test metadata down to exactly the 53 matches required by Kaggle
        test_meta_all.columns = [c.lower().strip() for c in test_meta_all.columns]
        sample_sub.columns = [c.lower().strip() for c in sample_sub.columns]
        
        test_meta = test_meta_all[test_meta_all['match_id'].isin(sample_sub['match_id'])].copy()
        test_meta = test_meta.set_index('match_id').reindex(sample_sub['match_id']).reset_index()
        
        num_test_rows = len(test_meta)
        if num_test_rows != 53:
            print(f"Warning: Found {num_test_rows} matches instead of 53. Proceeding with exact sample matching.")

        # 4. Generate real features
        X_train, y_train, X_test, label_encoder = build_cricket_features(train_df, test_meta)

        # 5. Ensemble Training & Probability Calibration
        print(f"Training ensemble on {X_train.shape[0]} valid historical matches...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            'xgb': xgb.XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=4, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, max_depth=4, random_state=42, verbose=-1),
            'cat': cb.CatBoostClassifier(iterations=250, learning_rate=0.04, depth=4, verbose=0, random_seed=42)
        }
        
        final_preds = np.zeros((len(X_test), 4))
        
        for name, model in models.items():
            print(f"Running Cross-Validation & Calibration for {name.upper()}...")
            clf = CalibratedClassifierCV(model, method='isotonic', cv=skf)
            clf.fit(X_train, y_train)
            final_preds += clf.predict_proba(X_test) / len(models)
            
        # 6. Build and Clean Submission Layout
        print("Writing final calibrated probabilities...")
        
        # Get standard class names mapped properly
        classes = list(label_encoder.classes_)
        sub = pd.DataFrame(final_preds, columns=classes)
        
        # Re-verify and map standard column layout uppercase variants if needed
        clean_prob_cols = []
        for orig_col in PROB_COLS:
            matched = [c for c in classes if c.lower() == orig_col.lower()]
            if matched:
                sub[orig_col] = sub[matched[0]]
            else:
                sub[orig_col] = 0.0
            clean_prob_cols.append(orig_col)
            
        sub = sub[clean_prob_cols]
        sub.insert(0, 'match_id', test_meta['match_id'].values)
        
        sub = normalize_probabilities(sub)
        sub.to_csv('submission_final.csv', index=False, float_format='%.4f')
        print(f"\nSUCCESS! Created 'submission_final.csv' with exactly {len(sub)} rows.")
        
    except Exception as e:
        print(f"\nPipeline execution broke: {e}")

if __name__ == "__main__":
    run_pipeline()