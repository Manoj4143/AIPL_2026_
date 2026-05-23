# ==============================================================================
# Final IPL Match Pipeline (53-Row Kaggle Submission Generator)
# ==============================================================================

import sys
import subprocess
import warnings
from pathlib import Path

# --- 1. Auto-Dependency Installer ---
def install_dependencies():
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost']
    print("Checking dependencies...")
    for package in required_packages:
        try:
            import_name = 'sklearn' if package == 'scikit-learn' else package
            __import__(import_name)
        except ImportError:
            print(f"Missing '{package}'. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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

# --- 4. Probability Normalization (Crucial for Log Loss) ---
def normalize_probabilities(preds_df):
    """Ensures absolute strict compliance with the sum-to-1.0 rule."""
    preds_df['A_small'] = preds_df['A_small'].round(4)
    preds_df['A_big'] = preds_df['A_big'].round(4)
    preds_df['B_small'] = preds_df['B_small'].round(4)
    
    # Force the last class to absorb rounding errors so row sum is EXACTLY 1.0
    preds_df['B_big'] = (1.0 - preds_df[['A_small', 'A_big', 'B_small']].sum(axis=1)).round(4)
    preds_df['B_big'] = preds_df['B_big'].clip(lower=0.0)
    return preds_df

# --- 5. Main Execution ---
def run_pipeline():
    print("Starting Final Pipeline...")
    
    try:
        # 1. READ THE SAMPLE SUBMISSION TO GET EXACT 53 MATCH IDs
        print("Reading sample_submission.csv for the 53 matches...")
        sample_sub = pd.read_csv('sample_submission.csv')
        test_match_ids = sample_sub['match_id'].values
        num_test_rows = len(test_match_ids)
        print(f"Found {num_test_rows} matches required for submission.")

        # 2. LOAD YOUR ACTUAL TRAINING DATA
        # Replace these lines with your actual feature engineering logic
        print("Loading training data (Replace with your actual engineered features)...")
        
        # NOTE: For this code to run out-of-the-box, it generates structural data. 
        # YOU MUST replace X_train with your actual processed train_IPL.csv data.
        X_train = pd.DataFrame(np.random.rand(500, 15), columns=[f'feat_{i}' for i in range(15)])
        y_train = pd.Series(np.random.choice(PROB_COLS, 500))
        
        # Generate X_test matching the EXACT number of rows needed (53)
        X_test = pd.DataFrame(np.random.rand(num_test_rows, 15), columns=[f'feat_{i}' for i in range(15)])

        # 3. TRAIN THE ENSEMBLE
        print(f"Training models on {X_train.shape[0]} matches...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            'xgb': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1),
            'cat': cb.CatBoostClassifier(iterations=200, learning_rate=0.05, depth=4, verbose=0, random_seed=42)
        }
        
        final_preds = np.zeros((num_test_rows, 4))
        
        for name, model in models.items():
            print(f"Training {name.upper()}...")
            clf = CalibratedClassifierCV(model, method='isotonic', cv=skf)
            clf.fit(X_train, y_encoded)
            final_preds += clf.predict_proba(X_test) / len(models)
            
        # 4. FORMAT SUBMISSION
        print("Formatting the exact 53-row output...")
        sub = pd.DataFrame(final_preds, columns=le.classes_)
        sub = sub[PROB_COLS] # Order columns correctly
        sub.insert(0, 'match_id', test_match_ids) # Insert the 53 match IDs
        
        sub = normalize_probabilities(sub)
        
        # 5. SAVE
        sub.to_csv('submission_final.csv', index=False, float_format='%.4f')
        print(f"\nSUCCESS! 'submission_final.csv' generated with EXACTLY {len(sub)} rows.")
        
    except FileNotFoundError:
        print("ERROR: Could not find 'sample_submission.csv'. Please ensure it is in the same folder as this script.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()