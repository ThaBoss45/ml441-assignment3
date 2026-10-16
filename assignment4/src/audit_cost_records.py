import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import FunctionTransformer


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def metrics(true, predicted, labels):
    require(np.isin(predicted, labels).all(), 'Unknown prediction label')
    precision, recall, score, support = precision_recall_fscore_support(true, predicted, labels=labels, zero_division=0)
    matrix = confusion_matrix(true, predicted, labels=labels)
    normal = np.asarray(true) == 0
    return {'labels': labels, 'zero_division': 0, 'macro_f1': float(score.mean()), 'balanced_accuracy': float(recall.mean()),
            'accuracy': float(np.mean(true == predicted)), 'normal_false_alarm_rate': float(np.mean(predicted[normal] != 0)),
            'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': score.tolist(), 'support': support.tolist(),
            'confusion': matrix.tolist(), 'row_normalised_confusion': np.divide(matrix, support[:, None], out=np.zeros(matrix.shape), where=support[:, None] != 0).tolist()}


def close(actual, expected, context):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), context + ': keys differ')
        for key, value in expected.items():
            close(actual[key], value, context + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), context + ': size differs')
        for index, value in enumerate(expected):
            close(actual[index], value, context + f'[{index}]')
    elif isinstance(expected, float):
        require(np.isfinite(actual) and abs(actual - expected) <= 1e-14, context + ': numerical disagreement')
    else:
        require(actual == expected, context + ': differs')


def assert_unscaled(encoder, frame):
    declared = next(transformer for name, transformer, columns in encoder.transformers if name == 'numeric')
    require(declared == 'passthrough', 'A scaler was configured in a tree pipeline')
    fitted = encoder.named_transformers_['numeric']
    require(isinstance(fitted, FunctionTransformer) and fitted.func is None, 'Tree numerical transformation is not identity')
    np.testing.assert_array_equal(np.asarray(fitted.transform(frame)), np.asarray(frame))


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'output/verification/cost_audits' / stamp
    output.mkdir(parents=True, exist_ok=False)
    hashed = {}

    def digest(path):
        path = Path(path).resolve()
        if path not in hashed:
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    value.update(block)
            hashed[path] = value.hexdigest()
        return hashed[path]

    try:
        protocol = read(root / 'configs/primary_protocol.json')
        spec = read(root / 'configs/comparable_cost_protocol.json')
        require(canonical(protocol) == (root / 'configs/primary_protocol.sha256').read_text().strip(), 'Primary protocol seal differs')
        require(canonical(spec) == (root / 'configs/comparable_cost_protocol.sha256').read_text().strip(), 'Cost protocol seal differs')
        for name, expected in spec['implementation'].items():
            require(digest(root / name) == expected, 'Cost runner source differs')
        directory = Path(next(iter(importlib.util.find_spec('ml441_assignment4').submodule_search_locations)))
        require({path.name: digest(path) for path in sorted(directory.glob('*.py'))} == protocol['implementation_code'], 'Frozen shared code differs')
        env = {'python': platform.python_version(), 'platform': platform.platform(), 'architecture': platform.machine(),
               'packages': {name: importlib.metadata.version(name) for name in protocol['execution_environment']['packages']}}
        require(env == protocol['execution_environment'], 'Frozen cost environment differs')
        for item in protocol['gate_evidence']:
            require(digest(root / item['path']) == item['sha256'], 'Sealed gate evidence differs')
        primary = root / 'output/primary' / canonical(protocol)[:16]
        cost = root / spec['output_root'] / canonical(spec)[:16]
        require(not (primary / 'process.lock').exists() and not (cost / 'process.lock').exists(), 'A model process lock remains')
        require(not (primary / 'failure.json').exists() and not (cost / 'failure.json').exists(), 'A model failure remains')
        require(digest(root / protocol['data']['file']) == protocol['data']['sha256'], 'Raw data identity differs')
        require(digest(root / protocol['splits']['source']) == spec['inputs']['split_sha256'], 'Historical split identity differs')
        require(digest(root / 'materials/assignment2_baseline_records/selected_parameters.json') == spec['inputs']['baseline_settings_sha256'], 'Historical controls differ')
        require(digest(root / 'materials/assignment2_baseline_records/outer_predictions.csv.gz') == spec['inputs']['baseline_predictions_sha256'], 'Historical prediction identity differs')
        raw = pd.read_csv(root / protocol['data']['file'], na_values=['?'], low_memory=False)
        target = raw[protocol['data']['target']].to_numpy(dtype=np.int8)
        features = raw.drop(columns=[protocol['data']['target'], *protocol['representation']['drop']]).copy()
        for column in protocol['representation']['nominal']:
            features[column] = features[column].astype('string').fillna('Missing')
        with np.load(root / protocol['splits']['source'], allow_pickle=False) as archive:
            partitions = {name: archive[name] for name in archive.files}
        historical = pd.read_csv(root / 'materials/assignment2_baseline_records/outer_predictions.csv.gz')
        old_settings = {row['outer_fold']: row for row in read(root / 'materials/assignment2_baseline_records/selected_parameters.json')}
        locked = {row['fold']: row for row in read(primary / 'selection_lock.json')['selections']}
        summary = read(cost / 'summary.json')
        require(summary['status'] == 'complete_exact_reproduction' and not summary['primary_scores_replaced'], 'Cost completion state differs')
        require(summary['cost_protocol_sha256'] == canonical(spec), 'Cost summary protocol differs')
        audit_paths = sorted((root / 'output/verification/primary_audits').glob('*/audit.json'))
        audit_path = next(path for path in reversed(audit_paths) if digest(path) == summary['complete_audit_sha256'])
        primary_audit = read(audit_path)
        require(primary_audit['status'] == 'complete_saved_record_audit_passed' and primary_audit['primary_comparison_ready_for_report'], 'Independent primary audit missing')
        for name in ['selection_lock.json', 'comparison_summary.json']:
            require(digest(primary / name) == primary_audit['verified_files_sha256'][str((primary / name).relative_to(root))], 'Audited primary evidence changed')
        labels = protocol['data']['class_order']
        expected_keys = {(fold, model, seed) for fold in spec['fold_order'] for model in spec['pipeline_order'] for seed in (spec['forest_seeds'] if model == 'forest' else [20260826 + fold])}
        summaries = {(row['identity']['fold'], row['identity']['model'], row['identity']['seed']): row for row in summary['records']}
        require(len(summary['records']) == len(summaries) == 25 and set(summaries) == expected_keys, 'Cost model coverage differs')
        manifests = sorted(cost.glob('fold_*/**/artifacts.json'))
        require(len(manifests) == 25, 'Cost manifest coverage differs')
        rows = []
        observations = []
        for count, manifest_path in enumerate(manifests, 1):
            folder = manifest_path.parent
            manifest = read(manifest_path)
            require(set(manifest) == {'record.json', 'predictions.npz', 'complete.json', 'model.joblib', 'preprocessor.joblib', 'representation.json', 'indices.npz'}, 'Cost manifest schema differs')
            for name, expected in manifest.items():
                require(digest(folder / name) == expected, 'Cost artifact differs: ' + str(folder / name))
            digest(manifest_path)
            record = read(folder / 'record.json')
            identity = record['identity']
            fold, model, seed = identity['fold'], identity['model'], identity['seed']
            key = (fold, model, seed)
            require(key in expected_keys and record == summaries[key], 'Cost summary and individual record differ')
            require(folder == cost / f'fold_{fold}' / model / f'seed_{seed}', 'Cost directory and identity differ')
            fitting, evaluation = [partitions[f'fold_{fold}_outer_{name}'] for name in ['train', 'test']]
            setting = locked[fold] if model == 'forest' else old_settings[fold]['classification_tree' if model == 'classification_tree' else 'knn']
            expected_identity = {'cost_protocol_sha256': canonical(spec), 'primary_protocol_sha256': canonical(protocol),
                                 'complete_audit_sha256': digest(audit_path), 'fold': fold, 'model': model, 'seed': seed, 'setting': setting,
                                 'fitting_sha256': array_hash(fitting), 'evaluation_sha256': array_hash(evaluation)}
            require(identity == expected_identity, 'Cost run identity differs')
            with np.load(folder / 'indices.npz', allow_pickle=False) as saved:
                require(np.array_equal(saved['fitting'], fitting) and np.array_equal(saved['evaluation'], evaluation), 'Cost row indices differ')
            if model == 'forest':
                reference = primary / 'evaluation' / f'fold_{fold}' / f'seed_{seed}'
                require(digest(reference / 'predictions.npz') == primary_audit['verified_files_sha256'][str((reference / 'predictions.npz').relative_to(root))], 'Primary reference predictions changed')
                with np.load(reference / 'predictions.npz', allow_pickle=False) as saved:
                    require(np.array_equal(saved['row_index'], evaluation), 'Primary reference rows differ')
                    expected = saved['predicted']
            else:
                name = 'Classification tree' if model == 'classification_tree' else 'kNN'
                reference_rows = historical[(historical.model == name) & (historical.outer_fold == fold)].sort_values('row_index')
                require(np.array_equal(reference_rows.row_index, evaluation), 'Historical reference rows differ')
                expected = reference_rows.predicted.to_numpy()
            with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
                require(np.array_equal(saved['row_index'], evaluation) and np.array_equal(saved['true'], target[evaluation]), 'Cost prediction rows or targets differ')
                require(np.array_equal(saved['expected'], expected) and np.array_equal(saved['predicted'], expected), 'Cost reproduction differs from original predictions')
                close(record['recalculated_metrics'], metrics(target[evaluation], saved['predicted'], labels), 'independent cost prediction metrics')
            require(record['prediction_disagreements'] == 0 and record['status'] == 'exact_prediction_reproduction', 'Cost agreement label differs')
            require(record['evaluation_rows'] == len(evaluation) and record['workers'] == spec['workers'] and record['timing_boundaries'] == spec['timing_boundaries'], 'Cost measurement scope differs')
            complete = read(folder / 'complete.json')
            require(complete['identity'] == identity and complete['identity_sha256'] == canonical(identity), 'Estimator identity differs')
            require(complete['model_sha256'] == manifest['model.joblib'], 'Estimator seal differs')
            require(complete['model_bytes'] == record['model_checkpoint_bytes'] == (folder / 'model.joblib').stat().st_size, 'Estimator size differs')
            require(record['encoder_bytes'] == (folder / 'preprocessor.joblib').stat().st_size and record['encoder_sha256'] == manifest['preprocessor.joblib'], 'Encoder size or seal differs')
            rep = read(folder / 'representation.json')
            require(canonical(rep) == record['representation_sha256'] and rep['fitting_index_sha256'] == array_hash(fitting), 'Representation identity differs')
            encoder = joblib.load(folder / 'preprocessor.joblib')
            require(encoder.feature_names_in_.tolist() == features.columns.tolist(), 'Encoder input schema differs')
            require(encoder.get_feature_names_out().tolist() == rep['output_columns'], 'Encoded coordinates differ')
            for column, levels in zip(rep['nominal_columns'], rep['categories'], strict=True):
                require(levels == sorted(features.iloc[fitting][column].unique().tolist()), 'Encoder categories are not fitting-only')
            if model == 'knn':
                scaler = encoder.named_transformers_['numeric']
                require(scaler.n_samples_seen_ == len(fitting) and not scaler.clip, 'Historical scaler fitting scope differs')
                numerical = features.iloc[fitting][rep['numeric_columns']].to_numpy()
                np.testing.assert_array_equal(scaler.data_min_, numerical.min(axis=0))
                np.testing.assert_array_equal(scaler.data_max_, numerical.max(axis=0))
            else:
                assert_unscaled(encoder, features.iloc[fitting[:25]][rep['numeric_columns']])
            payload = joblib.load(folder / 'model.joblib')
            require(payload['identity'] == identity, 'Embedded estimator identity differs')
            estimator = payload['payload']['estimator']
            require(estimator.n_features_in_ == len(rep['output_columns']), 'Estimator coordinate count differs')
            if model == 'forest':
                controls = {**protocol['ensemble']['fixed'], **setting['setting'], 'n_estimators': setting['trees'], 'random_state': seed, 'n_jobs': 1, 'verbose': 0, 'warm_start': False}
                require(estimator.get_params(deep=False) == controls, 'Forest estimator controls differ')
                original = read(reference / 'record.json')['costs']
                require([int(tree.random_state) for tree in estimator.estimators_] == original['tree_seeds'], 'Refit tree seeds differ')
                require([int(tree.tree_.max_depth) for tree in estimator.estimators_] == original['max_depths'], 'Refit tree depths differ')
                require(sum(tree.tree_.node_count for tree in estimator.estimators_) == original['total_nodes'], 'Refit forest node count differs')
            elif model == 'classification_tree':
                require(all(estimator.get_params()[name] == value for name, value in setting.items()) and estimator.random_state == seed, 'Historical tree controls differ')
                require(estimator.tree_.node_count == old_settings[fold]['tree_node_count'] and estimator.get_depth() == old_settings[fold]['tree_fitted_depth'], 'Historical tree structure differs')
            else:
                rng = np.random.default_rng(seed + 200)
                blocks = []
                for label in labels:
                    members = np.flatnonzero(target[fitting] == label)
                    blocks.append(np.sort(rng.choice(members, 5000, replace=False) if len(members) > 5000 else members))
                selected = np.concatenate(blocks)
                rng.shuffle(selected)
                require(np.array_equal(payload['payload']['prototype_positions'], selected), 'Neighbour prototype positions differ')
                require(np.array_equal(payload['payload']['prototype_labels'], target[fitting[selected]]), 'Neighbour prototype targets differ')
                require(record['prototype_row_index_sha256'] == array_hash(fitting[selected]) and record['prototype_rows'] == len(selected), 'Neighbour prototype record differs')
                require(estimator.n_neighbors == setting['k'] and estimator.metric == ('manhattan' if setting['p'] == 1 else 'euclidean') and estimator.n_jobs == 4, 'Neighbour estimator controls differ')
                rebuilt = sparse.csr_matrix(encoder.transform(features.iloc[fitting[selected]]), dtype=np.float32)
                rebuilt.sort_indices()
                require(estimator._fit_X.shape == rebuilt.shape and (estimator._fit_X != rebuilt).nnz == 0, 'Persisted neighbour fitting vectors differ')
            values = [record[name] for name in ['preprocessing_cost', 'fit_cost', 'prediction_cost', 'serialisation_cost']]
            for value in values:
                require(np.isfinite(value['elapsed_seconds']) and value['elapsed_seconds'] >= 0, 'Invalid measured time')
                require(value['limit_bytes'] == 6 * 1024 ** 3 and 0 < value['sampled_process_peak_rss_bytes'] <= value['limit_bytes'] and not value['limit_exceeded'], 'Cost memory limit differs')
                require(value['sampling_interval_seconds'] == 0.02, 'Cost memory sampling interval differs')
            utc_seconds = (datetime.fromisoformat(record['completed_utc']) - datetime.fromisoformat(record['started_utc'])).total_seconds()
            clock_gap = utc_seconds - record['run_wall_seconds']
            require(record['run_wall_seconds'] > 0 and utc_seconds >= 0, 'Invalid run duration')
            if abs(clock_gap) > 1:
                observations.append({'fold': fold, 'model': model, 'seed': seed, 'utc_interval_seconds': utc_seconds,
                                     'performance_counter_seconds': record['run_wall_seconds'], 'clock_gap_seconds': clock_gap,
                                     'interpretation': 'Report performance-counter elapsed time separately from UTC elapsed time; macOS counter excludes system sleep'})
            rows.append({'fold': fold, 'model': model, 'seed': seed, **{name + '_seconds': record[name]['elapsed_seconds'] for name in ['preprocessing_cost', 'fit_cost', 'prediction_cost', 'serialisation_cost']},
                         'sampled_process_peak_rss_bytes': max(value['sampled_process_peak_rss_bytes'] for value in values), 'model_checkpoint_bytes': record['model_checkpoint_bytes'],
                         'encoder_bytes': record['encoder_bytes'], 'prediction_disagreements': 0, 'counter_run_seconds': record['run_wall_seconds'], 'utc_interval_seconds': utc_seconds, 'clock_gap_seconds': clock_gap})
            del payload, estimator, encoder
            gc.collect()
            print(json.dumps({'verified_cost_models': count, 'planned': 25}), flush=True)
        pd.DataFrame(rows).sort_values(['fold', 'model', 'seed']).to_csv(output / 'costs_recalculated.csv', index=False)
        result = {'status': 'cost_saved_record_audit_passed_with_timing_scope_limitations', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'models_verified': len(rows), 'folds_verified': 5, 'prediction_disagreements': 0, 'cost_protocol_sha256': canonical(spec),
                  'complete_primary_audit': str(audit_path.relative_to(root)), 'summary_sha256': digest(cost / 'summary.json'), 'auditor_sha256': digest(__file__),
                  'clock_implementation': vars(time.get_clock_info('perf_counter')), 'timing_observations': observations,
                  'files_verified_sha256': {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): value for path, value in hashed.items()},
                  'estimator_controls_and_structure_checked': True, 'encoder_fit_scope_and_neighbour_vectors_checked': True,
                  'metrics_independently_recalculated_with_sklearn': True, 'models_refitted': 0, 'cost_predictions_replayed': False,
                  'primary_outputs_modified': False, 'cost_outputs_modified': False,
                  'limitations': ['Saved predictions and all artifact hashes checked; full distance searches and all tree predictions were not replayed',
                                  'Performance-counter durations are awake elapsed times on this macOS host, not full UTC intervals',
                                  'One cost run spans substantial system sleep; timing values are descriptive and not a continuously awake benchmark',
                                  'One timing observation per fold and algorithmic seed; no independent hardware replications or timing confidence intervals',
                                  'Sampled memory is whole-process RSS and storage includes the checkpoint identity wrapper']}
        (output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': result['status'], 'models': len(rows), 'timing_observations': len(observations), 'output': str(output)}), flush=True)
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
