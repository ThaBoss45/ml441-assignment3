import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import run_cost_comparison as COST


class CostComparisonTests(unittest.TestCase):
    def fixtures(self, root):
        spec = {'workers': {'forest_fit': 4, 'forest_prediction': 1, 'nearest_neighbour_search': 4, 'working_memory_mib': 256, 'native_threadpool_limit': 1},
                'budget': {'process_memory_gib': 6},
                'timing_boundaries': {phase: 'synthetic fixture' for phase in ['preprocessing', 'fit', 'prediction', 'serialisation']}}
        protocol = {'data': {'class_order': list(range(10))}, 'representation': {'nominal': ['proto', 'service', 'state']},
                    'ensemble': {'fixed': {'criterion': 'gini', 'bootstrap': True, 'max_samples': None, 'oob_score': False}}}
        target = np.tile(np.arange(10, dtype=np.int8), 8)
        features = pd.DataFrame({'signal': target.astype(float), 'proto': ['tcp'] * 80, 'service': ['Missing'] * 80, 'state': ['CON'] * 80})
        fitting, evaluation = np.arange(60), np.arange(60, 80)
        parts = {'fold_1_outer_train': fitting, 'fold_1_outer_test': evaluation}
        history = pd.concat([pd.DataFrame({'row_index': evaluation, 'outer_fold': 1, 'model': name, 'true': target[evaluation], 'predicted': target[evaluation]}) for name in ['Classification tree', 'kNN']], ignore_index=True)
        primary = root / 'primary'
        directory = primary / 'evaluation/fold_1/seed_11'
        directory.mkdir(parents=True)
        np.savez(directory / 'predictions.npz', row_index=evaluation, predicted=target[evaluation])
        audit = root / 'audit.json'
        audit.write_text('{}')
        return spec, protocol, target, features, parts, history, primary, audit

    def test_all_three_pipelines_persist_rows_costs_and_resume_without_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec, protocol, target, features, parts, history, primary, audit = self.fixtures(root)
            settings = {'classification_tree': {'criterion': 'gini', 'max_depth': 24, 'min_samples_leaf': 1, 'class_weight': 'balanced', 'ccp_alpha': 1e-5},
                        'knn': {'k': 3, 'p': 1, 'weights': 'distance', 'vote_exponent': 0.5},
                        'forest': {'setting': {'min_samples_leaf': 1, 'max_features': 1.0, 'class_weight': None}, 'trees': 6}}
            for model, setting in settings.items():
                seed = 11 if model == 'forest' else 20260827
                args = (root, root / 'cost', model, spec, protocol, primary, audit, features, target, parts, 1, seed, setting, history)
                result = COST.run_one(*args)
                self.assertGreater(result['fit_cost']['elapsed_seconds'], 0)
                self.assertGreater(result['prediction_cost']['elapsed_seconds'], 0)
                self.assertEqual(result['evaluation_rows'], 20)
                if model != 'forest':
                    self.assertEqual(result['status'], 'exact_prediction_reproduction')
                saved = root / 'cost/fold_1' / model / f'seed_{seed}'
                with np.load(saved / 'indices.npz', allow_pickle=False) as indices:
                    np.testing.assert_array_equal(indices['fitting'], np.arange(60))
                    np.testing.assert_array_equal(indices['evaluation'], np.arange(60, 80))
                with patch.object(COST, 'fit_model', side_effect=AssertionError('Resume must not refit')):
                    repeated = COST.run_one(*args)
                self.assertEqual(repeated, result)

    def test_reproduction_difference_is_retained_and_encoder_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            spec, protocol, target, features, parts, history, primary, audit = self.fixtures(root)
            history.loc[(history.model == 'kNN') & (history.row_index == 60), 'predicted'] = 1
            setting = {'k': 3, 'p': 1, 'weights': 'uniform', 'vote_exponent': 0.0}
            args = (root, root / 'cost', 'knn', spec, protocol, primary, audit, features, target, parts, 1, 20260827, setting, history)
            result = COST.run_one(*args)
            self.assertEqual(result['status'], 'reproduction_difference')
            self.assertEqual(result['prediction_disagreements'], 1)
            self.assertEqual(history.loc[(history.model == 'kNN') & (history.row_index == 60), 'predicted'].item(), 1)
            encoder = root / 'cost/fold_1/knn/seed_20260827/preprocessor.joblib'
            encoder.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact has changed'):
                COST.run_one(*args)


if __name__ == '__main__':
    unittest.main()
