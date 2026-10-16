import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml441_assignment4.primary import Study, validate_frozen_protocol
from ml441_assignment4.runtime_smoke import synthetic_inputs


def fixture_protocol():
    return {
        "data": {"class_order": list(range(10))},
        "budget": {"maximum_fit_hours": 1},
        "ensemble": {"fixed": {"criterion": "gini", "bootstrap": True, "max_samples": None, "oob_score": False},
                     "seeds": [11, 23, 37], "structural_n_estimators": 2,
                     "structural_grid": {"min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 1.0], "class_weight": [None, "balanced_subsample"]}},
        "representation": {"nominal": ["proto", "service", "state"]},
        "splits": {"outer_folds": 2},
        "selection": {"tree_count_stage": {"counts": [1, 2, 4]}},
    }


class PrimaryGate(unittest.TestCase):
    def test_unfrozen_protocol_cannot_start(self):
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            validate_frozen_protocol(Path('.'), {"status": "proposal_only_not_frozen"})

    def test_complete_selection_gate_and_compatible_resume(self):
        protocol, features, target, partitions = synthetic_inputs(fixture_protocol())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output/pilot/synthetic"
            study = Study(root, protocol, features, target, partitions, output, role="synthetic_test")
            with self.assertRaisesRegex(ValueError, "locked before outer"):
                study.evaluate()
            lock = study.select()
            self.assertEqual(len(lock["selections"]), 2)
            self.assertFalse((output / "evaluation").exists())
            self.assertEqual(len(list((output / "selection").rglob("record.json"))), 84)
            records = study.evaluate()
            self.assertEqual(len(records), 6)
            paths = sorted(output.rglob("model.joblib"))
            modification_times = [path.stat().st_mtime_ns for path in paths]
            resumed = Study(root, protocol, features, target, partitions, output, role="synthetic_test")
            with patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('Resume must not refit')), patch('sklearn.preprocessing.OneHotEncoder.fit', side_effect=AssertionError('Resume must preserve the fitted encoder')):
                resumed.evaluate()
            self.assertEqual(modification_times, [path.stat().st_mtime_ns for path in paths])
            for seed in protocol["ensemble"]["seeds"]:
                rows = []
                for fold in [1, 2]:
                    with np.load(output / "evaluation" / f"fold_{fold}" / f"seed_{seed}" / "predictions.npz", allow_pickle=False) as archive:
                        rows.extend(archive["row_index"].tolist())
                np.testing.assert_array_equal(np.sort(rows), np.arange(600))
            selected_path = output / "selection/fold_1/selected.json"
            selected_path.write_text(selected_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed after locking"):
                resumed.require_lock()


if __name__ == "__main__":
    unittest.main()
