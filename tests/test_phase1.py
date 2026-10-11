"""
Phase 1 Automated Test Suite for IPL Match Outcome Prediction System.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import (
    predict_match,
    validate_match_inputs,
    validate_probabilities,
    load_model_and_artifacts,
    get_available_teams,
    get_available_venues
)


class TestIPLOutcomePredictionPhase1(unittest.TestCase):

    def setUp(self):
        self.team_a = "Chennai Super Kings"
        self.team_b = "Gujarat Titans"
        self.venue = "MA Chidambaram Stadium"
        self.city = "Chennai"

    def test_01_valid_prediction(self):
        """Test a standard valid prediction with full parameters."""
        res = predict_match(
            team_a=self.team_a,
            team_b=self.team_b,
            venue=self.venue,
            city=self.city,
            toss_winner=self.team_a,
            toss_decision="bat"
        )
        self.assertIsInstance(res, dict)
        for cls_name in ['A_small', 'A_big', 'B_small', 'B_big']:
            self.assertIn(cls_name, res)
            self.assertGreaterEqual(res[cls_name], 0.0)
            self.assertLessEqual(res[cls_name], 1.0)

        # Confirm sum equals 1.0 within 1e-6
        prob_sum = res['probability_sum']
        self.assertAlmostEqual(prob_sum, 1.0, places=6)
        self.assertTrue(res['is_valid_sum'])
        self.assertIn(res['most_likely_outcome'], ['A_small', 'A_big', 'B_small', 'B_big'])

    def test_02_same_team_selection_rejected(self):
        """Test that selecting the same team for Team A and Team B raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            predict_match(
                team_a="Mumbai Indians",
                team_b="Mumbai Indians",
                venue="Wankhede Stadium",
                city="Mumbai",
                toss_winner="Mumbai Indians",
                toss_decision="field"
            )
        self.assertIn("cannot be the same team", str(ctx.exception))

    def test_03_missing_optional_data_handled(self):
        """Test that missing optional city or toss parameters does not crash."""
        res = predict_match(
            team_a="Royal Challengers Bengaluru",
            team_b="Kolkata Knight Riders",
            venue="M Chinnaswamy Stadium",
            city=None,
            toss_winner=None,
            toss_decision="field"
        )
        self.assertIsInstance(res, dict)
        self.assertTrue(res['is_valid_sum'])
        self.assertAlmostEqual(res['probability_sum'], 1.0, places=6)

    def test_04_missing_model_behavior(self):
        """Test that missing model files raise a clear FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model_and_artifacts(models_dir="/non/existent/models/directory", force_reload=True)
        self.assertIn("Trained CatBoost model not found", str(ctx.exception))

    def test_05_probability_sum_validation(self):
        """Test strict probability validation function."""
        valid_probs = {'A_small': 0.25, 'A_big': 0.25, 'B_small': 0.25, 'B_big': 0.25}
        self.assertTrue(validate_probabilities(valid_probs, tolerance=1e-6))

        invalid_sum_probs = {'A_small': 0.30, 'A_big': 0.30, 'B_small': 0.30, 'B_big': 0.30}
        with self.assertRaises(ValueError):
            validate_probabilities(invalid_sum_probs, tolerance=1e-6)

        out_of_bounds_probs = {'A_small': -0.1, 'A_big': 0.5, 'B_small': 0.3, 'B_big': 0.3}
        with self.assertRaises(ValueError):
            validate_probabilities(out_of_bounds_probs, tolerance=1e-6)

    def test_06_ui_classes_and_metadata(self):
        """Test that available teams, venues, and classes are non-empty."""
        teams = get_available_teams()
        venues = get_available_venues()
        self.assertGreater(len(teams), 5)
        self.assertGreater(len(venues), 5)
        self.assertIn("Chennai Super Kings", teams)
        self.assertIn("MA Chidambaram Stadium", venues)

    def test_07_invalid_toss_inputs(self):
        """Test that invalid toss winner or toss decision raises ValueError."""
        with self.assertRaises(ValueError):
            predict_match("Chennai Super Kings", "Gujarat Titans", "Wankhede Stadium", toss_winner="Invalid Team")

        with self.assertRaises(ValueError):
            predict_match("Chennai Super Kings", "Gujarat Titans", "Wankhede Stadium", toss_decision="invalid_decision")


if __name__ == '__main__':
    unittest.main()
