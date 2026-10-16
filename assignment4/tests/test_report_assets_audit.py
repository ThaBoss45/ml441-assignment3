import importlib.util
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location('report_asset_audit', Path(__file__).resolve().parents[1] / 'src/audit_report_assets.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReportAuditTests(unittest.TestCase):
    def test_absent_classes_have_zero_scores_and_fixed_denominator(self):
        matrix = np.zeros((10, 10), dtype=int)
        matrix[0, 0] = 1
        matrix[0, 1] = 1
        matrix[1, 1] = 2
        scores = AUDIT.assess(matrix)
        np.testing.assert_allclose(scores['precision'][:2], [1, 2 / 3])
        np.testing.assert_allclose(scores['recall'][:2], [.5, 1])
        np.testing.assert_allclose(scores['f1'][:2], [2 / 3, .8])
        self.assertAlmostEqual(scores['macro_f1'], (2 / 3 + .8) / 10)
        self.assertAlmostEqual(scores['balanced_accuracy'], .15)
        self.assertEqual(scores['normal_false_alarm_rate'], .5)
        self.assertEqual(int(scores['support'].sum()), 4)

    def test_nested_fold_summary_does_not_count_seeds_as_test_folds(self):
        values = {1: [0, 1, 2], 2: [2, 3, 4], 3: [4, 5, 6], 4: [6, 7, 8], 5: [8, 9, 10]}
        summary = AUDIT.nested(values)
        self.assertEqual(summary['mean'], 5)
        self.assertAlmostEqual(summary['fold_sd'], np.sqrt(10))
        self.assertNotAlmostEqual(summary['fold_sd'], np.std(sum(values.values(), []), ddof=1))
        with self.assertRaisesRegex(ValueError, 'Five folds'):
            AUDIT.nested({1: [1], 2: [2]})

    def test_pooled_score_differs_from_fold_mean_and_mean_precision_recall(self):
        first = np.zeros((10, 10), dtype=int)
        second = np.zeros((10, 10), dtype=int)
        first[0, 0], first[1, 0] = 90, 10
        second[0, 0], second[0, 1], second[1, 1] = 1, 9, 90
        a, b = AUDIT.assess(first), AUDIT.assess(second)
        pooled = AUDIT.assess(first + second)
        self.assertNotAlmostEqual(pooled['macro_f1'], (a['macro_f1'] + b['macro_f1']) / 2)
        average_p = (a['precision'][0] + b['precision'][0]) / 2
        average_r = (a['recall'][0] + b['recall'][0]) / 2
        self.assertNotAlmostEqual((a['f1'][0] + b['f1'][0]) / 2, 2 * average_p * average_r / (average_p + average_r))


if __name__ == '__main__':
    unittest.main()
