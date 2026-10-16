import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location('independent_audit', Path(__file__).resolve().parents[1] / 'src/audit_primary_records.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IndependentAuditTests(unittest.TestCase):
    def test_absent_classes_and_false_alarm_denominator(self):
        record = AUDIT.recalculate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [0, 1, 2])
        np.testing.assert_allclose(record['f1'], [2 / 3, 0.8, 0])
        self.assertAlmostEqual(record['macro_f1'], (2 / 3 + 0.8) / 3)
        self.assertEqual(record['balanced_accuracy'], 0.5)
        self.assertEqual(record['normal_false_alarm_rate'], 0.5)
        self.assertEqual(record['row_normalised_confusion'][2], [0, 0, 0])

    def test_pooled_score_is_not_fold_mean(self):
        first = AUDIT.recalculate(np.array([0, 1]), np.array([0, 0]), [0, 1])
        second = AUDIT.recalculate(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]), [0, 1])
        pooled = AUDIT.recalculate(np.array([0, 1, 0, 0, 0, 1]), np.array([0, 0, 0, 0, 0, 1]), [0, 1])
        self.assertNotAlmostEqual(pooled['macro_f1'], (first['macro_f1'] + second['macro_f1']) / 2)

    def test_invalid_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid target'):
            AUDIT.recalculate(np.array([0, 2]), np.array([0, 1]), [0, 1])

    def archive(self, folder, rows, predicted, probabilities):
        path = folder / 'predictions.npz'
        np.savez(path, row_index=rows, true=np.array([0, 1]), predicted=predicted, probabilities=probabilities, labels=np.array([0, 1]))
        return path

    def test_exact_argmax_tie_and_probability_row_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = self.archive(folder, [0, 1], [0, 1], [[0.5, 0.5], [0.1, 0.9]])
            record, _ = AUDIT.prediction_check(path, np.array([0, 1]), np.array([0, 1]), [0, 1])
            self.assertEqual(record['accuracy'], 1)
            path = self.archive(folder, [0, 1], [1, 1], [[0.5, 0.5], [0.1, 0.9]])
            with self.assertRaisesRegex(ValueError, 'argmax'):
                AUDIT.prediction_check(path, np.array([0, 1]), np.array([0, 1]), [0, 1])
            path = self.archive(folder, [0, 1], [0, 1], [[0.4, 0.4], [0.1, 0.9]])
            with self.assertRaisesRegex(ValueError, 'sum to one'):
                AUDIT.prediction_check(path, np.array([0, 1]), np.array([0, 1]), [0, 1])

    def test_rows_cannot_be_repeated_or_reordered(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.archive(Path(folder), [1, 0], [0, 1], [[0.8, 0.2], [0.1, 0.9]])
            with self.assertRaisesRegex(ValueError, 'row order'):
                AUDIT.prediction_check(path, np.array([0, 1]), np.array([0, 1]), [0, 1])
            path = self.archive(Path(folder), [0, 0], [0, 1], [[0.8, 0.2], [0.1, 0.9]])
            with self.assertRaisesRegex(ValueError, 'Repeated'):
                AUDIT.prediction_check(path, np.array([0, 0]), np.array([0, 1]), [0, 1])

    def test_tolerance_does_not_chain_and_seeds_cannot_be_selected(self):
        rows = []
        for identifier, score, nodes in [('a', 0.8, 300), ('b', 0.8 - 0.75e-12, 200), ('c', 0.8 - 1.5e-12, 100)]:
            for seed in [11, 23, 37]:
                rows.append({'candidate_id': identifier, 'seed': seed, 'macro_f1': score, 'balanced_accuracy': 0.7, 'total_nodes': nodes, 'trees': 100})
        self.assertEqual(AUDIT.choose(rows, [11, 23, 37], 1e-12)['candidate_id'], 'b')
        with self.assertRaisesRegex(ValueError, 'selection seeds'):
            AUDIT.choose(rows[:-1], [11, 23, 37], 1e-12)

    def test_manifest_tampering_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            payload = folder / 'record.json'
            payload.write_text('{}')
            (folder / 'artifacts.json').write_text(json.dumps({'record.json': 'bad-hash'}))
            audit = AUDIT.Audit.__new__(AUDIT.Audit)
            audit.hashed = {}
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                audit.manifest(folder, 'artifacts.json')
            (folder / 'artifacts.json').write_text(json.dumps({'../other': 'bad-hash'}))
            with self.assertRaisesRegex(ValueError, 'leaves its directory'):
                audit.manifest(folder, 'artifacts.json')

    def test_complete_status_requires_finished_process_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaisesRegex(ValueError, 'five-fold selection lock'):
                AUDIT.require_complete(folder)
            (folder / 'selection_lock.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'completed outer summaries'):
                AUDIT.require_complete(folder)
            (folder / 'comparison_summary.json').write_text('{}')
            (folder / 'process.lock').write_text('1')
            with self.assertRaisesRegex(ValueError, 'stopped'):
                AUDIT.require_complete(folder)
            (folder / 'process.lock').unlink()
            AUDIT.require_complete(folder)

    def test_complete_comparison_rebuild_rejects_tripled_seed_support(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'primary'
            directory.mkdir()
            target = np.tile(np.array([0, 1], dtype=np.int8), 5)
            mapping = np.repeat(np.arange(1, 6), 2)
            metric = {'labels': [0, 1], 'zero_division': 0, 'macro_f1': 1.0, 'balanced_accuracy': 1.0, 'accuracy': 1.0,
                      'normal_false_alarm_rate': 0.0, 'precision': [1.0, 1.0], 'recall': [1.0, 1.0], 'f1': [1.0, 1.0],
                      'support': [1, 1], 'confusion': [[1, 0], [0, 1]], 'row_normalised_confusion': [[1.0, 0.0], [0.0, 1.0]]}
            pooled = {**metric, 'support': [5, 5], 'confusion': [[5, 0], [0, 5]]}
            seeds = [11, 23, 37]
            outer, records, seed_rows = [], {}, []
            for seed in seeds:
                for fold in range(1, 6):
                    location = directory / 'evaluation' / f'fold_{fold}' / f'seed_{seed}'
                    location.mkdir(parents=True)
                    rows = np.flatnonzero(mapping == fold)
                    np.savez(location / 'predictions.npz', row_index=rows, predicted=target[rows])
                    records[('outer', fold, 'candidate', seed, 100)] = {'metrics': metric, 'checkpoint_bytes': 123}
                    outer.append({'fold': fold, 'seed': seed, **metric, 'checkpoint_bytes': 123})
                    seed_rows.append({'fold': fold, 'seed': seed, **metric})
            history = root / 'materials/assignment2_baseline_records/outer_predictions.csv.gz'
            history.parent.mkdir(parents=True)
            pd.concat([pd.DataFrame({'model': name, 'row_index': np.arange(10), 'true': target, 'predicted': target, 'outer_fold': mapping}) for name in ['Classification tree', 'kNN']]).to_csv(history, index=False)
            folds = [{'fold': f, 'macro_f1_seed_mean': 1.0, 'macro_f1_seed_sd': 0.0} for f in range(1, 6)]
            baseline = {name: {'folds': {str(f): metric for f in range(1, 6)}, 'pooled': pooled} for name in ['Classification tree', 'kNN']}
            paired = {name: [{'fold': f, 'ensemble_seed_mean_macro_f1': 1.0, 'comparator_macro_f1': 1.0, 'difference': 0.0} for f in range(1, 6)] for name in baseline}
            summary = {'status': 'verified', 'evidence_role': 'synthetic audit fixture', 'folds': folds, 'seed_fold_records': seed_rows,
                       'macro_f1_fold_mean': 1.0, 'macro_f1_fold_sd': 0.0, 'pooled_by_seed': {str(seed): pooled for seed in seeds},
                       'baselines': baseline, 'paired_comparisons': paired, 'inference': 'Descriptive paired folds; no independence or formal significance claim'}
            (directory / 'comparison_summary.json').write_text(json.dumps(summary))
            (directory / 'outer_records.json').write_text(json.dumps(outer))
            audit = AUDIT.Audit.__new__(AUDIT.Audit)
            audit.root, audit.directory, audit.target, audit.mapping, audit.labels, audit.records, audit.hashed = root, directory, target, mapping, np.array([0, 1]), records, {}
            audit.protocol = {'ensemble': {'seeds': seeds}, 'comparators': {'prediction_sha256': AUDIT.digest(history)}, 'evidence_role': 'synthetic audit fixture'}
            result = audit.complete_summary()
            self.assertEqual(result['independent_summary']['pooled_by_seed']['11']['support'], [5, 5])
            summary['pooled_by_seed']['11']['support'] = [15, 15]
            (directory / 'comparison_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'numerical disagreement|differs'):
                audit.complete_summary()


if __name__ == '__main__':
    unittest.main()
