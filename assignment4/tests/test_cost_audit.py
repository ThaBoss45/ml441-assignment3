import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


SPEC = importlib.util.spec_from_file_location('cost_audit', Path(__file__).resolve().parents[1] / 'src/audit_cost_records.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CostAuditTests(unittest.TestCase):
    def test_fitted_passthrough_is_accepted_and_scaler_is_rejected(self):
        frame = pd.DataFrame({'value': [10.0, 40.0, 70.0]})
        identity = ColumnTransformer([('numeric', 'passthrough', ['value'])]).fit(frame)
        AUDIT.assert_unscaled(identity, frame)
        scaled = ColumnTransformer([('numeric', MinMaxScaler(), ['value'])]).fit(frame)
        with self.assertRaisesRegex(ValueError, 'scaler was configured'):
            AUDIT.assert_unscaled(scaled, frame)

    def test_independent_metrics_keep_fixed_absent_labels(self):
        value = AUDIT.metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [0, 1, 2])
        np.testing.assert_allclose(value['f1'], [2 / 3, 0.8, 0])
        self.assertEqual(value['normal_false_alarm_rate'], 0.5)
        self.assertEqual(value['support'], [2, 2, 0])
        with self.assertRaisesRegex(ValueError, 'Unknown prediction'):
            AUDIT.metrics(np.array([0, 1]), np.array([0, 3]), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
