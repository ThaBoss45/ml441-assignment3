import importlib.util
import copy
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location('report_asset_audit', Path(__file__).resolve().parents[1] / 'src/audit_report_assets.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReportAuditTests(unittest.TestCase):
    def test_screening_projection_uses_attack_population_and_retains_type_errors(self):
        matrix = np.zeros((10, 10), dtype=int)
        matrix[0, 0], matrix[0, 1] = 80, 20
        matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[2, 2] = 5, 10, 15, 20
        report = AUDIT.screening_counts(matrix)
        self.assertEqual(report['normal_support'], 100)
        self.assertEqual(report['attack_support'], 50)
        self.assertEqual(report['false_alarm_count'], 20)
        self.assertEqual(report['missed_attack_count'], 5)
        self.assertEqual(report['normal_false_alarm_rate'], .2)
        self.assertEqual(report['attack_miss_rate'], .1)
        self.assertNotAlmostEqual(report['attack_miss_rate'], 1 - AUDIT.assess(matrix)['precision'][0])
        self.assertEqual(int(matrix[1:, 1:].sum()), 45)
        self.assertEqual(int(matrix.trace()) - matrix[0, 0], 30)
        with self.assertRaisesRegex(ValueError, 'both be represented'):
            AUDIT.screening_counts(np.diag([1] + [0] * 9))

    def test_hypothetical_cost_threshold_has_correct_orientation(self):
        ensemble = {'false_alarm_count': 10, 'missed_attack_count': 6}
        baseline = {'false_alarm_count': 30, 'missed_attack_count': 2}
        ratio = AUDIT.screening_break_even(ensemble, baseline)
        self.assertEqual(ratio, 5)
        self.assertLess(10 + 4 * 6, 30 + 4 * 2)
        self.assertEqual(10 + ratio * 6, 30 + ratio * 2)
        self.assertGreater(10 + 6 * 6, 30 + 6 * 2)
        with self.assertRaisesRegex(ValueError, 'trade-off absent'):
            AUDIT.screening_break_even({'false_alarm_count': 5, 'missed_attack_count': 1}, baseline)

    def test_confusion_labels_reject_transposed_cells_and_incorrect_units(self):
        matrix = np.eye(10)
        matrix[0, 0], matrix[0, 3] = .424, .576
        labels = [{'true': row, 'predicted': row, 'proportion': float(matrix[row, row]),
                   'text': '42' if row == 0 else '100'} for row in range(10)]
        labels.append({'true': 0, 'predicted': 3, 'proportion': .576, 'text': '58'})
        AUDIT.verify_confusion_annotations(matrix, labels)
        wrong = copy.deepcopy(labels)
        wrong[-1]['true'], wrong[-1]['predicted'] = 3, 0
        with self.assertRaisesRegex(ValueError, 'coverage'):
            AUDIT.verify_confusion_annotations(matrix, wrong)
        wrong = copy.deepcopy(labels)
        wrong[-1]['text'] = '0.576'
        with self.assertRaisesRegex(ValueError, 'percentage'):
            AUDIT.verify_confusion_annotations(matrix, wrong)
        wrong = copy.deepcopy(labels)
        wrong[0]['proportion'] = .426
        with self.assertRaisesRegex(ValueError, 'value'):
            AUDIT.verify_confusion_annotations(matrix, wrong)

    def test_confusion_labels_resolve_equal_errors_and_preserve_small_positive_values(self):
        matrix = np.eye(10)
        matrix[0, 0], matrix[0, 1], matrix[0, 2] = .992, .004, .004
        labels = [{'true': row, 'predicted': row, 'proportion': float(matrix[row, row]),
                   'text': str(round(float(100 * matrix[row, row])))} for row in range(10)]
        labels.append({'true': 0, 'predicted': 1, 'proportion': .004, 'text': '<1'})
        AUDIT.verify_confusion_annotations(matrix, labels)
        for alteration in ['tied_class', 'rounded_to_zero', 'duplicate']:
            wrong = copy.deepcopy(labels)
            if alteration == 'tied_class':
                wrong[-1]['predicted'] = 2
            elif alteration == 'rounded_to_zero':
                wrong[-1]['text'] = '0'
            else:
                wrong.append(copy.deepcopy(wrong[-1]))
            with self.assertRaises(ValueError):
                AUDIT.verify_confusion_annotations(matrix, wrong)

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
