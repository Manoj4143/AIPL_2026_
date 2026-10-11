"""
Inference and Prediction Service for IPL Match Outcome Prediction.

Exposes predict_match(...) and supporting validation logic.
Separates UI presentation from ML execution.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import catboost as cb

from src.preprocessing import (
    TEAM_MAPPING,
    TARGET_CLASSES,
    FEATURE_COLS,
    find_data_file
)

_MODEL_CACHE = {}


def get_models_directory():
    """
    Finds the directory containing serialized models across standard project locations.
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_dir = os.path.dirname(src_dir)

    candidates = [
        os.path.join(project_dir, 'models'),
        os.path.join(project_dir, 'AIPL_2026_', 'models'),
        os.path.join(project_dir, '..', 'models'),
        os.path.join(project_dir, '..', 'AIPL_2026_', 'models'),
        os.path.join('.', 'models'),
        os.path.join('.', 'AIPL_2026_', 'models')
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, 'catboost_model.cbm')):
            return os.path.abspath(c)
    return os.path.join(project_dir, 'models')


def load_model_and_artifacts(models_dir=None, force_reload=False):
    """
    Loads and caches the CatBoost model and metadata.
    Raises FileNotFoundError with clear instructions if model is missing.
    """
    global _MODEL_CACHE
    if not force_reload and 'model' in _MODEL_CACHE:
        return _MODEL_CACHE['model'], _MODEL_CACHE['artifacts']

    if models_dir is None:
        models_dir = get_models_directory()

    model_path = os.path.join(models_dir, 'catboost_model.cbm')
    artifacts_path = os.path.join(models_dir, 'model_artifacts.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained CatBoost model not found at '{model_path}'. "
            f"Please run 'python -m src.preprocessing' to train and export the model file."
        )

    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"Model artifacts file not found at '{artifacts_path}'. "
            f"Please run 'python -m src.preprocessing' to generate artifacts."
        )

    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    artifacts = joblib.load(artifacts_path)

    # Ensure venue_matches exists in artifacts metadata
    if 'venue_matches' not in artifacts.get('metadata', {}):
        metadata_json_path = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(metadata_json_path):
            try:
                import json
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    jmeta = json.load(f)
                if 'venue_matches' in jmeta:
                    artifacts['metadata']['venue_matches'] = jmeta['venue_matches']
            except Exception:
                pass

    _MODEL_CACHE['model'] = model
    _MODEL_CACHE['artifacts'] = artifacts
    return model, artifacts


def get_metadata():
    """
    Returns metadata including available teams, venues, cities, and toss options.
    """
    _, artifacts = load_model_and_artifacts()
    return artifacts.get('metadata', {})


def get_available_teams():
    """
    Returns sorted list of recognized team names.
    """
    meta = get_metadata()
    return meta.get('all_teams', [])


def get_available_venues():
    """
    Returns sorted list of recognized venues.
    """
    meta = get_metadata()
    return meta.get('all_venues', [])


def get_venue_city_map():
    """
    Returns dictionary mapping venues to cities.
    """
    meta = get_metadata()
    return meta.get('venue_to_city', {})


def validate_match_inputs(team_a, team_b, venue, city, toss_winner, toss_decision):
    """
    Validates user inputs before prediction.
    Raises ValueError on invalid inputs.
    """
    if not team_a or not str(team_a).strip():
        raise ValueError("Team A must be provided.")
    if not team_b or not str(team_b).strip():
        raise ValueError("Team B must be provided.")

    clean_a = TEAM_MAPPING.get(team_a.strip(), team_a.strip())
    clean_b = TEAM_MAPPING.get(team_b.strip(), team_b.strip())

    if clean_a == clean_b:
        raise ValueError(
            f"Team A and Team B cannot be the same team ('{clean_a}'). "
            "Please select two different teams."
        )

    if not toss_winner or not str(toss_winner).strip():
        raise ValueError("Toss winner must be selected.")

    clean_toss_winner = TEAM_MAPPING.get(toss_winner.strip(), toss_winner.strip())
    if clean_toss_winner not in [clean_a, clean_b]:
        raise ValueError(
            f"Toss winner ('{clean_toss_winner}') must be either Team A ('{clean_a}') "
            f"or Team B ('{clean_b}')."
        )

    if not toss_decision or not str(toss_decision).strip():
        raise ValueError("Toss decision must be specified.")

    clean_decision = toss_decision.strip().lower()
    if clean_decision not in ['bat', 'field']:
        raise ValueError(
            f"Invalid toss decision '{toss_decision}'. Toss decision must be either 'bat' or 'field'."
        )

    if not venue or not str(venue).strip():
        raise ValueError("Venue must be specified.")

    return clean_a, clean_b, venue.strip(), (city.strip() if city else None), clean_toss_winner, clean_decision


def build_inference_features(bat_first, bat_second, venue, toss_winner, toss_decision, artifacts):
    """
    Constructs the feature vector for a match using project feature definitions.
    """
    metadata = artifacts['metadata']
    team_stats = metadata['team_stats']
    venue_profile = metadata['venue_profile']
    cat_encoders = artifacts['cat_encoders']

    team_a_win_rate = team_stats.get(bat_first, {}).get('win_rate', 0.5)
    team_b_win_rate = team_stats.get(bat_second, {}).get('win_rate', 0.5)
    team_a_bat1st_rate = team_stats.get(bat_first, {}).get('bat_first_win_rate', 0.5)
    canonical_venue = canonicalize_venue_name(venue, metadata.get('all_venues'))
    venue_avg_score = venue_profile.get(canonical_venue, 330.0)
    win_rate_diff = team_a_win_rate - team_b_win_rate

    # Safe categorical encoding
    def safe_encode(encoder, val):
        if val in set(encoder.classes_):
            return int(encoder.transform([val])[0])
        return -1

    bat_first_enc = safe_encode(cat_encoders['Bat First'], bat_first)
    bat_second_enc = safe_encode(cat_encoders['Bat Second'], bat_second)
    venue_enc = safe_encode(cat_encoders['Venue'], canonical_venue)
    toss_decision_enc = safe_encode(cat_encoders['toss_decision'], toss_decision)
    team_a_won_toss = 1 if toss_winner == bat_first else 0

    feature_dict = {
        'team_a_win_rate': team_a_win_rate,
        'team_b_win_rate': team_b_win_rate,
        'team_a_bat1st_rate': team_a_bat1st_rate,
        'venue_avg_score': venue_avg_score,
        'win_rate_diff': win_rate_diff,
        'Bat First_enc': bat_first_enc,
        'Bat Second_enc': bat_second_enc,
        'Venue_enc': venue_enc,
        'toss_decision_enc': toss_decision_enc,
        'team_a_won_toss': team_a_won_toss
    }

    # Verify all required feature columns exist
    missing_cols = [c for c in FEATURE_COLS if c not in feature_dict]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    df_feats = pd.DataFrame([feature_dict])[FEATURE_COLS]
    return df_feats


def validate_probabilities(probabilities, tolerance=1e-6):
    """
    Validates that predicted probabilities:
      1. Are all between 0 and 1.
      2. Sum to 1.0 within tolerance.
    """
    for cls_name, prob in probabilities.items():
        if prob < 0.0 or prob > 1.0:
            raise ValueError(
                f"Prediction validation failed: class '{cls_name}' probability {prob} "
                "is outside the valid range [0, 1]."
            )

    prob_sum = sum(probabilities.values())
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError(
            f"Prediction validation failed: sum of probabilities is {prob_sum:.8f}, "
            f"which deviates from 1.0 by more than tolerance {tolerance}."
        )
    return True


def predict_match(team_a, team_b, venue, city=None, toss_winner=None, toss_decision='field'):
    """
    Main prediction service function.

    Parameters:
      team_a (str): Name of Team A
      team_b (str): Name of Team B
      venue (str): Match venue
      city (str, optional): Host city
      toss_winner (str, optional): Team winning the toss (defaults to team_a)
      toss_decision (str, optional): 'bat' or 'field' (defaults to 'field')

    Returns:
      dict: Dictionary with keys:
        'A_small', 'A_big', 'B_small', 'B_big',
        'most_likely_outcome', 'most_likely_probability',
        'batting_first', 'batting_second',
        'probability_sum', 'is_valid_sum'
    """
    # Default toss_winner if not provided
    if toss_winner is None:
        toss_winner = team_a

    # 1. Validate inputs
    clean_a, clean_b, clean_venue, clean_city, clean_toss_winner, clean_decision = validate_match_inputs(
        team_a, team_b, venue, city, toss_winner, toss_decision
    )

    # 2. Determine innings batting order from toss dynamics
    if clean_decision == 'bat':
        bat_first = clean_toss_winner
        bat_second = clean_b if clean_toss_winner == clean_a else clean_a
    else:  # 'field'
        bat_second = clean_toss_winner
        bat_first = clean_b if clean_toss_winner == clean_a else clean_a

    # 3. Load model and artifacts
    model, artifacts = load_model_and_artifacts()

    # 4. Build feature input
    X = build_inference_features(
        bat_first, bat_second, clean_venue, clean_toss_winner, clean_decision, artifacts
    )

    # 5. Predict probabilities
    try:
        raw_probas = model.predict_proba(X)[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed during model inference: {str(e)}") from e

    le_target = artifacts['le_target']
    class_order = list(le_target.classes_)
    prob_map = {cls: float(raw_probas[i]) for i, cls in enumerate(class_order)}

    # Model's internal outputs:
    # A_big: Bat First wins big
    # A_small: Bat First wins small
    # B_big: Bat Second wins big
    # B_small: Bat Second wins small
    bf_small = prob_map.get('A_small', 0.25)
    bf_big = prob_map.get('A_big', 0.25)
    bs_small = prob_map.get('B_small', 0.25)
    bs_big = prob_map.get('B_big', 0.25)

    # 6. Map back to user's Team A and Team B
    if bat_first == clean_a:
        p_A_small = bf_small
        p_A_big = bf_big
        p_B_small = bs_small
        p_B_big = bs_big
    else:
        # Team B batted first
        p_A_small = bs_small
        p_A_big = bs_big
        p_B_small = bf_small
        p_B_big = bf_big

    # Normalization check and strict sum-to-1 enforcement
    raw_dict = {
        'A_small': p_A_small,
        'A_big': p_A_big,
        'B_small': p_B_small,
        'B_big': p_B_big
    }
    total = sum(raw_dict.values())
    if total > 0:
        normalized = {k: v / total for k, v in raw_dict.items()}
    else:
        normalized = {k: 0.25 for k in raw_dict}

    # Validate output probabilities
    validate_probabilities(normalized, tolerance=1e-6)

    # Identify most-likely outcome
    most_likely = max(normalized, key=normalized.get)
    max_prob = normalized[most_likely]

    result = {
        'A_small': normalized['A_small'],
        'A_big': normalized['A_big'],
        'B_small': normalized['B_small'],
        'B_big': normalized['B_big'],
        'most_likely_outcome': most_likely,
        'most_likely_probability': max_prob,
        'team_a': clean_a,
        'team_b': clean_b,
        'venue': clean_venue,
        'city': clean_city,
        'toss_winner': clean_toss_winner,
        'toss_decision': clean_decision,
        'batting_first': bat_first,
        'batting_second': bat_second,
        'probability_sum': sum(normalized.values()),
        'is_valid_sum': abs(sum(normalized.values()) - 1.0) <= 1e-6
    }
    return result


def canonicalize_venue_name(venue, known_venues=None):
    """
    Canonicalizes a venue string to match historical training metadata names.
    Normalizes whitespace, repeated spaces, punctuation variations, and casing.
    Returns the exact matching canonical name from known_venues if found;
    otherwise returns the cleaned venue string.
    """
    if not venue or not str(venue).strip():
        return ""

    clean_venue = re.sub(r'\s+', ' ', str(venue).strip())

    if known_venues is None:
        try:
            meta = get_metadata()
            known_venues = meta.get('all_venues', [])
        except Exception:
            known_venues = []

    if not known_venues:
        return clean_venue

    # 1. Exact match
    if clean_venue in known_venues:
        return clean_venue

    # 2. Case-insensitive exact match
    lower_map = {v.lower(): v for v in known_venues}
    if clean_venue.lower() in lower_map:
        return lower_map[clean_venue.lower()]

    # 3. Normalized punctuation & casing match
    def simplify(s):
        s_clean = s.replace('.', '').replace(',', '').replace('-', ' ').replace("'", '')
        s_clean = re.sub(r'\s+', ' ', s_clean).strip().lower()
        # Collapse separated initials like "m a " -> "ma "
        s_clean = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', s_clean)
        return s_clean

    simplified_map = {simplify(v): v for v in known_venues}
    venue_simplified = simplify(clean_venue)
    if venue_simplified in simplified_map:
        return simplified_map[venue_simplified]

    # Check without spaces for edge cases
    no_space_map = {simplify(v).replace(' ', ''): v for v in known_venues}
    venue_no_space = venue_simplified.replace(' ', '')
    if venue_no_space in no_space_map:
        return no_space_map[venue_no_space]

    # Substring matching for full official names vs colloquial names
    for v in known_venues:
        v_simp = simplify(v)
        if venue_simplified == v_simp or (len(venue_simplified) > 10 and venue_simplified in v_simp) or (len(v_simp) > 10 and v_simp in venue_simplified):
            return v

    return clean_venue


# Genuine historical match counts from IPL training dataset (2008-2024)
HISTORICAL_VENUE_MATCH_COUNTS = {
    "Wankhede Stadium": 123,
    "Eden Gardens": 98,
    "M Chinnaswamy Stadium": 98,
    "MA Chidambaram Stadium": 91,
    "Rajiv Gandhi International Stadium": 82,
    "Sawai Mansingh Stadium": 61,
    "Punjab Cricket Association Stadium": 61,
    "Feroz Shah Kotla": 60,
    "Maharashtra Cricket Association Stadium": 51,
    "Dubai International Cricket Stadium": 46,
    "Dr DY Patil Sports Academy": 37,
    "Arun Jaitley Stadium": 34,
    "Sheikh Zayed Stadium": 29,
    "Sharjah Cricket Stadium": 28,
    "Narendra Modi Stadium": 28,
    "Brabourne Stadium": 27,
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": 19,
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": 17,
    "Kingsmead": 15,
    "Himachal Pradesh Cricket Association Stadium": 13,
    "Sardar Patel Stadium": 12,
    "SuperSport Park": 12,
    "Saurashtra Cricket Association Stadium": 10,
    "Holkar Cricket Stadium": 9,
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": 9,
    "New Wanderers Stadium": 8,
    "Zayed Cricket Stadium": 8,
    "JSCA International Stadium Complex": 7,
    "Barabati Stadium": 7,
    "St George's Park": 7,
    "Newlands": 7,
    "Shaheed Veer Narayan Singh International Stadium": 6,
    "Nehru Stadium": 5,
    "Barsapara Cricket Stadium": 5,
    "Green Park": 4,
    "De Beers Diamond Oval": 3,
    "Vidarbha Cricket Association Stadium": 3,
    "Buffalo Park": 3,
    "OUTsurance Oval": 2
}


def get_venue_details(venue_name):
    """
    Returns verified historical statistics and profiles for a venue.
    Canonicalizes the venue name before looking up genuine training metadata.
    For unknown venues, returns a documented safe fallback clearly marked as estimated.
    """
    meta = get_metadata()
    venue_profile = meta.get('venue_profile', {})
    venue_to_city = meta.get('venue_to_city', {})
    venue_matches = meta.get('venue_matches') or HISTORICAL_VENUE_MATCH_COUNTS
    known_venues = meta.get('all_venues', list(venue_profile.keys()))

    clean_venue = str(venue_name).strip() if venue_name else ""
    canonical = canonicalize_venue_name(clean_venue, known_venues)
    is_known = canonical in venue_profile

    if is_known:
        avg_score = float(venue_profile[canonical])
        city = venue_to_city.get(canonical, "Unknown / Not Specified")
        matches = int(venue_matches.get(canonical, HISTORICAL_VENUE_MATCH_COUNTS.get(canonical, 0)))
        is_high_scoring = avg_score > 330.0
        scoring_type = "High-Scoring Ground" if is_high_scoring else "Balanced / Bowler-Friendly"
        status = "Historical Data Available"
    else:
        avg_score = 330.0
        city = venue_to_city.get(clean_venue, "Unknown / Not Specified")
        matches = 0
        is_high_scoring = False
        scoring_type = "Estimated Baseline (No Historical Training Matches)"
        status = "Estimated / Ground Not in Training Dataset"

    return {
        'venue': canonical if is_known else clean_venue,
        'canonical_name': canonical,
        'original_name': clean_venue,
        'city': city,
        'avg_score': avg_score,
        'match_count': matches,
        'is_high_scoring': is_high_scoring,
        'scoring_type': scoring_type,
        'is_known': is_known,
        'status': status
    }


# Standard alias for compatibility
get_venue_insights = get_venue_details


def get_team_comparison(team_a, team_b):
    """
    Returns genuine historical comparison statistics for Team A and Team B.
    """
    meta = get_metadata()
    team_stats = meta.get('team_stats', {})

    clean_a = TEAM_MAPPING.get(str(team_a).strip(), str(team_a).strip())
    clean_b = TEAM_MAPPING.get(str(team_b).strip(), str(team_b).strip())

    stats_a = team_stats.get(clean_a, {'win_rate': 0.5, 'bat_first_win_rate': 0.5, 'total_matches': 0})
    stats_b = team_stats.get(clean_b, {'win_rate': 0.5, 'bat_first_win_rate': 0.5, 'total_matches': 0})

    rate_a = float(stats_a.get('win_rate', 0.5))
    rate_b = float(stats_b.get('win_rate', 0.5))
    bat1_a = float(stats_a.get('bat_first_win_rate', 0.5))
    bat1_b = float(stats_b.get('bat_first_win_rate', 0.5))
    matches_a = int(stats_a.get('total_matches', 0))
    matches_b = int(stats_b.get('total_matches', 0))

    diff = rate_a - rate_b

    return {
        'team_a': clean_a,
        'team_b': clean_b,
        'team_a_win_rate': rate_a,
        'team_b_win_rate': rate_b,
        'team_a_bat1st_rate': bat1_a,
        'team_b_bat1st_rate': bat1_b,
        'team_a_matches': matches_a,
        'team_b_matches': matches_b,
        'win_rate_diff': diff
    }


def get_prediction_explanation(team_a, team_b, venue, toss_winner, toss_decision, result=None):
    """
    Computes exact TreeSHAP attributions and human-readable feature summaries for the prediction.
    Uses CatBoost's native C++ TreeSHAP implementation without requiring external dependencies.
    """
    if result is None:
        result = predict_match(team_a, team_b, venue, toss_winner=toss_winner, toss_decision=toss_decision)

    model, artifacts = load_model_and_artifacts()
    clean_a = result['team_a']
    clean_b = result['team_b']
    bat_first = result['batting_first']
    bat_second = result['batting_second']
    clean_venue = result['venue']
    clean_decision = result['toss_decision']
    clean_winner = result['toss_winner']
    most_likely = result['most_likely_outcome']

    X = build_inference_features(bat_first, bat_second, clean_venue, clean_winner, clean_decision, artifacts)
    pool = cb.Pool(X)
    shap_vals = model.get_feature_importance(pool, type='ShapValues')

    le_target = artifacts['le_target']
    classes = list(le_target.classes_)

    # Map user outcome back to model internal class
    if bat_first == clean_a:
        internal_target = most_likely
    else:
        swap = {'A_small': 'B_small', 'A_big': 'B_big', 'B_small': 'A_small', 'B_big': 'A_big'}
        internal_target = swap[most_likely]

    target_idx = classes.index(internal_target)
    raw_shap = shap_vals[0, target_idx, :-1]
    base_val = float(shap_vals[0, target_idx, -1])

    readable_labels = {
        'team_a_win_rate': f"{bat_first} Win Rate",
        'team_b_win_rate': f"{bat_second} Win Rate",
        'team_a_bat1st_rate': f"{bat_first} Bat-First Win Rate",
        'venue_avg_score': "Venue Scoring Baseline",
        'win_rate_diff': "Strength Differential",
        'Bat First_enc': f"{bat_first} Identity Factor",
        'Bat Second_enc': f"{bat_second} Identity Factor",
        'Venue_enc': "Venue Specific Factor",
        'toss_decision_enc': "Toss Decision Influence",
        'team_a_won_toss': "Toss Advantage"
    }

    feature_contributions = []
    for feat_name, sv in zip(FEATURE_COLS, raw_shap):
        feature_contributions.append({
            'feature': feat_name,
            'label': readable_labels.get(feat_name, feat_name),
            'value': float(X[feat_name].iloc[0]),
            'attribution': float(sv)
        })

    # Sort contributions by absolute magnitude
    feature_contributions.sort(key=lambda x: abs(x['attribution']), reverse=True)

    # Human-readable summary points
    summary_points = []
    comp = get_team_comparison(clean_a, clean_b)
    if comp['win_rate_diff'] > 0.04:
        summary_points.append(
            f"**Historical Win Rate Advantage**: {clean_a} holds a higher win rate ({comp['team_a_win_rate']*100:.1f}%) "
            f"than {clean_b} ({comp['team_b_win_rate']*100:.1f}%)."
        )
    elif comp['win_rate_diff'] < -0.04:
        summary_points.append(
            f"**Historical Win Rate Advantage**: {clean_b} holds a higher win rate ({comp['team_b_win_rate']*100:.1f}%) "
            f"than {clean_a} ({comp['team_a_win_rate']*100:.1f}%)."
        )
    else:
        summary_points.append(
            f"**Closely Matched Teams**: Both sides possess comparable historical win rates "
            f"({comp['team_a_win_rate']*100:.1f}% vs {comp['team_b_win_rate']*100:.1f}%)."
        )

    v_details = get_venue_details(clean_venue)
    summary_points.append(
        f"**Venue Profile**: {clean_venue} averages {v_details['avg_score']:.1f} total match runs ({v_details['scoring_type']})."
    )

    summary_points.append(
        f"**Toss & Innings Strategy**: {clean_winner} chose to {clean_decision}. "
        f"**{bat_first}** will bat first setting a target; **{bat_second}** will chase."
    )

    return {
        'most_likely_outcome': most_likely,
        'base_value': base_val,
        'feature_contributions': feature_contributions,
        'summary_points': summary_points,
        'batting_first': bat_first,
        'batting_second': bat_second,
        'team_a': clean_a,
        'team_b': clean_b,
        'venue': clean_venue
    }


def manage_prediction_history(history_list, new_entry=None, clear=False):
    """
    Manages in-memory session prediction history list.
    """
    if clear:
        return []
    history = list(history_list) if history_list is not None else []
    if new_entry:
        history.insert(0, new_entry)
    return history

