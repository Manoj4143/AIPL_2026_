"""
Phase 2 Automated Test Suite for IPL Match Outcome Prediction System.
Extends the testing suite with:
  - Input-to-feature mapping
  - Toss-order dynamics mapping
  - Correct Team A and Team B output labels
  - Model explanation data generation (TreeSHAP)
  - Team comparison generation
  - Venue metadata retrieval & missing metadata behavior
  - Prediction history management & clearing
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import (
    predict_match,
    validate_match_inputs,
    validate_probabilities,
    load_model_and_artifacts,
    build_inference_features,
    get_venue_details,
    get_team_comparison,
    get_prediction_explanation,
    manage_prediction_history,
    get_available_teams,
    get_available_venues
)


class TestIPLOutcomePredictionPhase2(unittest.TestCase):

    def setUp(self):
        self.team_a = "Chennai Super Kings"
        self.team_b = "Gujarat Titans"
        self.venue = "MA Chidambaram Stadium"
        self.city = "Chennai"

    # --- 1. Toss-Order Dynamics and Team Output Mapping ---
    def test_01_toss_bat_first_team_a(self):
        """When Team A wins toss and bats, Team A must bat first."""
        res = predict_match(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            city=self.city,
            toss_winner=self.team_a,
            toss_decision="bat"
        )
        self.assertEqual(res['batting_first'], self.team_a)
        self.assertEqual(res['batting_second'], self.team_b)
        self.assertTrue(res['is_valid_sum'])

    def test_02_toss_field_team_a(self):
        """When Team A wins toss and fields, Team B must bat first."""
        res = predict_match(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            city=self.city,
            toss_winner=self.team_a,
            toss_decision="field"
        )
        self.assertEqual(res['batting_first'], self.team_b)
        self.assertEqual(res['batting_second'], self.team_a)
        self.assertTrue(res['is_valid_sum'])

    def test_03_toss_bat_team_b(self):
        """When Team B wins toss and bats, Team B must bat first."""
        res = predict_match(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            city=self.city,
            toss_winner=self.team_b,
            toss_decision="bat"
        )
        self.assertEqual(res['batting_first'], self.team_b)
        self.assertEqual(res['batting_second'], self.team_a)

    def test_04_toss_field_team_b(self):
        """When Team B wins toss and fields, Team A must bat first."""
        res = predict_match(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            city=self.city,
            toss_winner=self.team_b,
            toss_decision="field"
        )
        self.assertEqual(res['batting_first'], self.team_a)
        self.assertEqual(res['batting_second'], self.team_b)

    # --- 2. Input-to-Feature Mapping ---
    def test_05_input_to_feature_mapping(self):
        """Verify feature matrix construction preserves feature names and expected dimensions."""
        _, artifacts = load_model_and_artifacts()
        X = build_inference_features(
            bat_first=self.team_a,
            bat_second=self.team_b,
            venue=self.venue,
            toss_winner=self.team_a,
            toss_decision="bat",
            artifacts=artifacts
        )
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(X.shape, (1, 10))
        expected_cols = [
            'team_a_win_rate', 'team_b_win_rate', 'team_a_bat1st_rate', 'venue_avg_score',
            'win_rate_diff', 'Bat First_enc', 'Bat Second_enc', 'Venue_enc', 'toss_decision_enc', 'team_a_won_toss'
        ]
        self.assertEqual(list(X.columns), expected_cols)
        self.assertAlmostEqual(X['win_rate_diff'].iloc[0], X['team_a_win_rate'].iloc[0] - X['team_b_win_rate'].iloc[0])
        self.assertEqual(X['team_a_won_toss'].iloc[0], 1)

    # --- 3. Model Explanation (TreeSHAP) ---
    def test_06_explanation_data_generation(self):
        """Verify prediction explanation generates SHAP contributions and summary points."""
        expl = get_prediction_explanation(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            toss_winner=self.team_a,
            toss_decision="bat"
        )
        self.assertIn('most_likely_outcome', expl)
        self.assertIn('feature_contributions', expl)
        self.assertIn('summary_points', expl)
        self.assertEqual(len(expl['feature_contributions']), 10)
        self.assertGreater(len(expl['summary_points']), 0)
        for contrib in expl['feature_contributions']:
            self.assertIn('feature', contrib)
            self.assertIn('label', contrib)
            self.assertIn('attribution', contrib)
            self.assertIsInstance(contrib['attribution'], float)

    # --- 4. Team Comparison Generation ---
    def test_07_team_comparison_generation(self):
        """Verify team comparison outputs genuine win rates and differentials."""
        comp = get_team_comparison(self.team_a, self.team_b)
        self.assertIn('team_a_win_rate', comp)
        self.assertIn('team_b_win_rate', comp)
        self.assertIn('team_a_bat1st_rate', comp)
        self.assertIn('team_b_bat1st_rate', comp)
        self.assertIn('win_rate_diff', comp)
        self.assertGreater(comp['team_a_matches'], 0)
        self.assertGreater(comp['team_b_matches'], 0)
        self.assertAlmostEqual(comp['win_rate_diff'], comp['team_a_win_rate'] - comp['team_b_win_rate'])

    # --- 5. Venue Metadata & Missing Metadata Handling ---
    def test_08_venue_metadata_display(self):
        """Verify venue details return genuine average scores and match counts."""
        v_info = get_venue_details(self.venue)
        self.assertEqual(v_info['venue'], self.venue)
        self.assertEqual(v_info['city'], "Chennai")
        self.assertGreater(v_info['avg_score'], 200.0)
        self.assertGreater(v_info['match_count'], 0)
        self.assertIn(v_info['scoring_type'], ["High-Scoring Ground", "Balanced / Bowler-Friendly"])

    def test_09_missing_venue_metadata_fallback(self):
        """Verify that an unseen venue falls back safely to default priors without crashing."""
        v_info = get_venue_details("Brand New Unknown Ground")
        self.assertEqual(v_info['venue'], "Brand New Unknown Ground")
        self.assertEqual(v_info['city'], "Unknown / Not Specified")
        self.assertEqual(v_info['avg_score'], 330.0)
        self.assertEqual(v_info['match_count'], 0)

    # --- 6. Prediction History Management ---
    def test_10_prediction_history_management(self):
        """Verify session history appends and clears properly."""
        history = []
        entry1 = {'Team A': 'CSK', 'Team B': 'GT', 'Outcome': 'A_small'}
        entry2 = {'Team A': 'MI', 'Team B': 'RCB', 'Outcome': 'B_big'}

        history = manage_prediction_history(history, new_entry=entry1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['Team A'], 'CSK')

        history = manage_prediction_history(history, new_entry=entry2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['Team A'], 'MI')

        # Test clear
        cleared = manage_prediction_history(history, clear=True)
        self.assertEqual(len(cleared), 0)


if __name__ == '__main__':
    unittest.main()
