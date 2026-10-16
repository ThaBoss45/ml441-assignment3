import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml441_assignment4.data import corrected_groups
from ml441_assignment4.experiment import unknown_category_check
from ml441_assignment4.metrics import classification_metrics, paired_fold_differences
from ml441_assignment4.models import diversity_record, probability_aggregate, probability_labels
from ml441_assignment4.preprocessing import forest_matrix, make_preprocessor
from ml441_assignment4.records import load_checkpoint, save_checkpoint
from ml441_assignment4.search import select_across_seeds, structural_candidates
from ml441_assignment4.splits import assert_separated
from ml441_assignment4.verify_results import verify_prediction_archive


class FixedMember:
    def __init__(self, probabilities):
        self.classes_ = np.array([0, 1])
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, matrix):
        return np.tile(self.probabilities, (matrix.shape[0], 1))


class ExperimentContract(unittest.TestCase):
    def test_groups_exclude_row_position_and_do_not_need_target(self):
        features = pd.DataFrame({"value": [1, 1, 2], "service": ["a", "a", "b"]}, index=[5, 9, 14])
        groups = corrected_groups(features)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        with self.assertRaisesRegex(ValueError, "group overlap"):
            assert_separated([0], [1], groups)
        assert_separated([0, 1], [2], groups)

    def test_partition_rejects_repeated_or_overlapping_rows(self):
        groups = np.arange(4)
        with self.assertRaisesRegex(ValueError, "Repeated row"):
            assert_separated([0, 0], [1], groups)
        with self.assertRaisesRegex(ValueError, "Row overlap"):
            assert_separated([0, 1], [1, 2], groups)

    def test_encoder_uses_only_fit_levels_and_ignores_unknown(self):
        fitting = pd.DataFrame({"number": [2.0, 5.0], "proto": ["tcp", "udp"], "service": ["http", "Missing"], "state": ["A", "A"]})
        nominal = ["proto", "service", "state"]
        preprocessor = make_preprocessor(fitting, nominal)
        encoded = forest_matrix(preprocessor.fit_transform(fitting), fitting=True)
        self.assertEqual(encoded.format, "csc")
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(unknown_category_check(preprocessor, fitting, nominal)["status"], "passed")
        unseen = fitting.iloc[:1].copy()
        unseen["proto"] = "validation_only"
        before = [values.copy() for values in preprocessor.named_transformers_["categorical"].categories_]
        preprocessor.transform(unseen)
        for old, new in zip(before, preprocessor.named_transformers_["categorical"].categories_):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(encoded.toarray()[:, 0], fitting.number)

    def test_probability_averaging_differs_from_majority_vote(self):
        forest = SimpleNamespace(classes_=np.array([2, 8]), estimators_=[FixedMember([0.51, 0.49]), FixedMember([0.51, 0.49]), FixedMember([0, 1])])
        probabilities, hard = probability_aggregate(forest, np.zeros((1, 1)), [2, 8])
        np.testing.assert_allclose(probabilities, [[0.34, 0.66]])
        self.assertEqual(probability_labels(probabilities, [2, 8])[0], 8)
        diversity, vote = diversity_record(hard, [2, 8])
        self.assertEqual(vote[0], 2)
        self.assertAlmostEqual(diversity["mean_pairwise_disagreement"], 2 / 3)

    def test_exact_class_tie_uses_smallest_sorted_label(self):
        np.testing.assert_array_equal(probability_labels([[0.5, 0.5]], [2, 8]), [2])
        with self.assertRaises(ValueError):
            probability_labels([[0.5, 0.5]], [8, 2])

    def test_real_forest_member_encoding_is_aligned_to_original_classes(self):
        matrix = np.arange(40, dtype=np.float32).reshape(-1, 1)
        target = np.where(matrix[:, 0] < 20, 2, 8)
        forest = RandomForestClassifier(n_estimators=5, min_samples_leaf=3, random_state=11, n_jobs=1).fit(matrix, target)
        probabilities, _ = probability_aggregate(forest, matrix, [2, 8])
        np.testing.assert_allclose(probabilities, forest.predict_proba(matrix), atol=1e-14, rtol=0)

    def test_balanced_subsample_weights_and_leaf_counts(self):
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(400, 4)).astype(np.float32)
        target = np.array([0] * 270 + [1] * 90 + [2] * 40)
        forest = RandomForestClassifier(n_estimators=3, min_samples_leaf=7, class_weight="balanced_subsample", random_state=11, n_jobs=1).fit(matrix, target)
        for tree, bag in zip(forest.estimators_, forest.estimators_samples_):
            self.assertEqual(len(bag), len(target))
            self.assertLess(len(np.unique(bag)), len(bag))
            self.assertEqual(tree.tree_.n_node_samples[0], len(np.unique(bag)))
            multiplicity = np.bincount(bag, minlength=len(target))
            counts = np.bincount(target[bag], minlength=3)
            weights = multiplicity * len(bag) / (3 * counts[target])
            np.testing.assert_allclose(tree.tree_.value[0, 0], [1 / 3] * 3, atol=1e-14, rtol=0)
            assigned = tree.apply(matrix)
            for leaf in np.flatnonzero(tree.tree_.children_left == -1):
                self.assertGreaterEqual(tree.tree_.n_node_samples[leaf], 7)
                rows = assigned == leaf
                weighted_class = np.bincount(target[rows], weights=weights[rows], minlength=3)
                np.testing.assert_allclose(tree.tree_.value[leaf, 0], weighted_class / weighted_class.sum(), atol=1e-14, rtol=0)

    def test_weighted_warm_start_and_prefix_match_direct_fit(self):
        rng = np.random.default_rng(13)
        matrix = rng.normal(size=(300, 6)).astype(np.float32)
        target = np.array([0] * 180 + [1] * 80 + [2] * 40)
        for weight in [None, "balanced_subsample"]:
            common = dict(random_state=23, class_weight=weight, min_samples_leaf=3, n_jobs=1)
            direct = RandomForestClassifier(n_estimators=8, **common).fit(matrix, target)
            staged = RandomForestClassifier(n_estimators=4, warm_start=True, **common).fit(matrix, target)
            prefix = staged.predict_proba(matrix)
            staged.set_params(n_estimators=8).fit(matrix, target)
            for first, second in zip(direct.estimators_, staged.estimators_):
                np.testing.assert_array_equal(first.tree_.__getstate__()["nodes"], second.tree_.__getstate__()["nodes"])
                np.testing.assert_array_equal(first.tree_.value, second.tree_.value)
            np.testing.assert_array_equal(staged.predict_proba(matrix), direct.predict_proba(matrix))
            aggregate, _ = probability_aggregate(direct, matrix, [0, 1, 2], count=4)
            np.testing.assert_array_equal(prefix, aggregate)

    def test_fixed_label_metrics_include_absent_classes(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 2])
        np.testing.assert_allclose(result["f1"], [2 / 3, 0.8, 0])
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["normal_false_alarm_rate"], 0.5)
        np.testing.assert_array_equal(result["row_normalised_confusion"][2], [0, 0, 0])

    def test_selection_uses_all_seeds_and_global_tolerance_set(self):
        records = []
        for candidate, score, nodes in [("a", 0.8, 200), ("b", 0.8 - 0.75e-12, 100), ("c", 0.8 - 1.5e-12, 10)]:
            for seed in [11, 23, 37]:
                records.append({"candidate_id": candidate, "seed": seed, "macro_f1": score, "balanced_accuracy": 0.7, "total_nodes": nodes, "trees": 100})
        self.assertEqual(select_across_seeds(records, [11, 23, 37])["candidate_id"], "b")
        with self.assertRaisesRegex(ValueError, "every fixed seed"):
            select_across_seeds(records[:-1], [11, 23, 37])
        with self.assertRaises(ValueError):
            select_across_seeds(records + [records[0]], [11, 23, 37])

    def test_structural_grid_is_twelve_unique_candidates(self):
        protocol = {"ensemble": {"structural_grid": {"min_samples_leaf": [1, 5, 25], "max_features": ["sqrt", 1.0], "class_weight": [None, "balanced_subsample"]}}}
        candidates = structural_candidates(protocol)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(set(row["candidate_id"] for row in candidates)), 12)

    def test_checkpoint_rejects_changed_identity_and_corruption(self):
        identity = {"data": "hash", "seed": 11, "indices": [1, 3]}
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(temporary, identity, {"value": np.arange(3)})
            np.testing.assert_array_equal(load_checkpoint(temporary, identity)["value"], [0, 1, 2])
            for changed in [{**identity, "seed": 23}, {**identity, "indices": [3, 1]}, {**identity, "data": "other"}]:
                with self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
                    load_checkpoint(temporary, changed)
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "content hash"):
                load_checkpoint(temporary, identity)

    def test_saved_prediction_verifier_rejects_swapped_rows(self):
        target = np.array([0, 1, 0])
        rows = np.array([0, 2])
        probability = np.array([[0.8, 0.2], [0.3, 0.7]])
        record = classification_metrics(target[rows], [0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            np.savez(path, row_index=rows, true=target[rows], predicted=[0, 1], probabilities=probability)
            self.assertEqual(verify_prediction_archive(path, rows, target, [0, 1], record)["status"], "passed")
            np.savez(path, row_index=rows[::-1], true=target[rows], predicted=[0, 1], probabilities=probability)
            with self.assertRaisesRegex(ValueError, "rows or targets"):
                verify_prediction_archive(path, rows, target, [0, 1], record)

    def test_seed_fits_are_averaged_within_matched_fold(self):
        result = paired_fold_differences([
            {"fold": fold, "seed": seed, "macro_f1": score}
            for fold, score in [(1, 0.8), (2, 0.6)] for seed in [11, 23, 37]
        ], {1: 0.7, 2: 0.55}, [11, 23, 37])
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose([row["difference"] for row in result], [0.1, 0.05])


if __name__ == "__main__":
    unittest.main()
