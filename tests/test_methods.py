import unittest

import numpy as np
import pandas as pd

from airway_methods.features import SOURCE_COLUMNS, VOICE, add_relative_features
from airway_methods.models import CLINICAL, SCHEMES, estimator, evaluate


class FeatureTests(unittest.TestCase):
    def test_directions_and_missingness(self):
        frame = pd.DataFrame({col: [1., 1.] for col in SOURCE_COLUMNS})
        frame.loc[0, "i_lpc_F3"] = -2
        frame.loc[0, "i_mid_high_energy_ratio"] = 4
        frame.loc[0, "i_cpps_full"] = 0
        frame.loc[1, "i_lpc_F1"] = np.nan
        result = add_relative_features(frame)
        self.assertEqual(result.loc[0, VOICE[0]], 3)
        self.assertEqual(result.loc[0, VOICE[1]], 3)
        self.assertEqual(result.loc[0, VOICE[6]], -1)
        self.assertTrue(np.isnan(result.loc[1, VOICE[7]]))
        self.assertEqual(len(VOICE), 8)

    def test_missing_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing per-vowel"):
            add_relative_features(pd.DataFrame({"age": [1]}))


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(43)
        size = 48
        frame = pd.DataFrame({"patient_id": [f"local-{i}" for i in range(size)],
                              "target": [0, 1] * (size // 2)})
        for column in set(CLINICAL) | set(VOICE) | {"age"}:
            frame[column] = rng.normal(size=size)
        frame.loc[0, "clinical_bmi"] = np.nan
        self.frame = frame

    def test_historical_input_counts(self):
        self.assertEqual([len(SCHEMES[str(i)]) for i in range(1, 5)], [10, 9, 19, 13])
        self.assertEqual(estimator("2", "rf", 3).estimator.named_steps["model"].n_estimators, 260)
        self.assertEqual(estimator("4", "extra", 3).estimator.named_steps["model"].n_estimators, 400)

    def test_aggregate_only_and_unique_patients(self):
        result = evaluate(self.frame, "1", folds=3)
        self.assertEqual((result["n"], result["events"]), (48, 24))
        self.assertEqual(result["tn"] + result["fp"] + result["fn"] + result["tp"], 48)
        self.assertNotIn("patient_id", result)
        bad = self.frame.copy()
        bad.loc[1, "patient_id"] = bad.loc[0, "patient_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate(bad, "1", folds=3)


if __name__ == "__main__":
    unittest.main()
