import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ordered_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def equal(actual, expected, context='value'):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{context}: keys differ')
        for key, value in expected.items():
            equal(actual[key], value, f'{context}.{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{context}: list size differs')
        for i, value in enumerate(expected):
            equal(actual[i], value, f'{context}[{i}]')
    elif isinstance(expected, float):
        require(np.isfinite(actual) and abs(actual - expected) <= 1e-14, f'{context}: numerical disagreement')
    else:
        require(actual == expected, f'{context}: differs')


def recalculate(true, predicted, labels):
    true, predicted, labels = np.asarray(true), np.asarray(predicted), np.asarray(labels)
    require(np.array_equal(labels, np.arange(len(labels))), 'Audit requires fixed contiguous labels beginning at zero')
    require(true.ndim == 1 and true.shape == predicted.shape, 'Prediction lengths differ')
    require(np.isin(true, labels).all() and np.isin(predicted, labels).all(), 'Invalid target or prediction label')
    matrix = np.bincount((len(labels) * true + predicted).astype(np.int64), minlength=len(labels) ** 2).reshape(len(labels), len(labels))
    actual, guessed, correct = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    precision = np.array([correct[c] / guessed[c] if guessed[c] else 0.0 for c in labels])
    recall = np.array([correct[c] / actual[c] if actual[c] else 0.0 for c in labels])
    scores = np.array([2 * correct[c] / (actual[c] + guessed[c]) if actual[c] + guessed[c] else 0.0 for c in labels])
    normalised = [matrix[c].astype(float) / actual[c] if actual[c] else np.zeros(len(labels)) for c in labels]
    return {'labels': labels.tolist(), 'zero_division': 0, 'macro_f1': float(scores.mean()),
            'balanced_accuracy': float(recall.mean()), 'accuracy': float(correct.sum() / len(true)) if len(true) else 0.0,
            'normal_false_alarm_rate': float((actual[0] - correct[0]) / actual[0]) if actual[0] else None,
            'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': scores.tolist(), 'support': actual.tolist(),
            'confusion': matrix.tolist(), 'row_normalised_confusion': np.asarray(normalised).tolist()}


def prediction_check(path, rows, target, labels):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'row_index', 'true', 'predicted', 'probabilities', 'labels'}, 'Prediction archive schema differs')
        saved_rows, true, predicted, probability, saved_labels = [archive[k] for k in ['row_index', 'true', 'predicted', 'probabilities', 'labels']]
    require(np.array_equal(saved_rows, rows), 'Evaluation row order differs')
    require(len(np.unique(saved_rows)) == len(rows), 'Repeated evaluation row')
    require(np.array_equal(true, target[rows]), 'Evaluation targets differ')
    require(np.array_equal(saved_labels, labels), 'Probability class order differs')
    require(probability.shape == (len(rows), len(labels)), 'Probability dimensions differ')
    require(np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all(), 'Invalid probability value')
    require(np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-12), 'Probability rows do not sum to one')
    require(np.array_equal(np.asarray(labels)[probability.argmax(axis=1)], predicted), 'Prediction violates sorted-class exact argmax')
    return recalculate(true, predicted, labels), predicted


def candidates(protocol):
    grid = protocol['ensemble']['structural_grid']
    result = {}
    for leaf, feature, weight in itertools.product(grid['min_samples_leaf'], grid['max_features'], grid['class_weight']):
        key = f"leaf{leaf:03d}_features_{'sqrt' if feature == 'sqrt' else 'all'}_weight_{weight or 'none'}"
        result[key] = {'min_samples_leaf': leaf, 'max_features': feature, 'class_weight': weight}
    return result


def choose(rows, seeds, tolerance, count_stage=False):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['candidate_id'], []).append(row)
    require(bool(grouped), 'No selection candidates')
    means = []
    for key, runs in grouped.items():
        require(sorted(row['seed'] for row in runs) == sorted(seeds), 'Incomplete or duplicate selection seeds')
        means.append({'candidate_id': key, **{field: float(np.mean([row[field] for row in runs])) for field in ['macro_f1', 'balanced_accuracy', 'total_nodes', 'trees']}})
    require(all(np.isfinite(row[field]) for row in means for field in ['macro_f1', 'balanced_accuracy', 'total_nodes', 'trees']), 'Nonfinite selection input')
    top = max(row['macro_f1'] for row in means)
    eligible = [row for row in means if top - row['macro_f1'] <= tolerance]
    recall = max(row['balanced_accuracy'] for row in eligible)
    eligible = [row for row in eligible if recall - row['balanced_accuracy'] <= tolerance]
    cost = 'trees' if count_stage else 'total_nodes'
    return sorted(eligible, key=lambda row: (row[cost], row['candidate_id']))[0]


def summary_row(record, identifier=None):
    identity = record['identity']
    return {'candidate_id': identifier or identity['candidate_id'], 'seed': identity['seed'], 'trees': identity['trees'],
            'total_nodes': record['costs']['total_nodes'], 'macro_f1': record['metrics']['macro_f1'],
            'balanced_accuracy': record['metrics']['balanced_accuracy']}


def require_complete(directory):
    require((directory / 'selection_lock.json').is_file(), 'Complete audit requires the five-fold selection lock')
    require((directory / 'comparison_summary.json').is_file(), 'Complete audit requires completed outer summaries')
    require(not (directory / 'process.lock').exists(), 'Complete audit requires the primary process to have stopped')
    require(not (directory / 'failure.json').exists(), 'Primary failure record requires review')


class Audit:
    def __init__(self, root, mode):
        self.root = Path(root).resolve()
        self.mode = mode
        self.protocol = read(self.root / 'configs/primary_protocol.json')
        self.seal = (self.root / 'configs/primary_protocol.sha256').read_text().strip()
        require(canonical(self.protocol) == self.seal, 'Protocol seal differs')
        self.directory = self.root / 'output/primary' / self.seal[:16]
        if mode == 'complete':
            require_complete(self.directory)
        self.lock_at_start = (self.directory / 'selection_lock.json').exists()
        self.selected_paths = sorted(self.directory.glob('selection/fold_*/selected.json'))
        self.snapshot = sorted(self.directory.glob('selection/**/artifacts.json'))
        if self.lock_at_start:
            self.snapshot += sorted(self.directory.glob('evaluation/**/artifacts.json'))
        self.hashed = {}
        self.prepared = {}
        self.records = {}
        self.regenerated = {}
        self.started = datetime.now(timezone.utc).isoformat()

    def hashed_file(self, path):
        path = Path(path).resolve()
        if path not in self.hashed:
            self.hashed[path] = digest(path)
        return self.hashed[path]

    def manifest(self, folder, name):
        folder = Path(folder).resolve()
        values = read(folder / name)
        for relative, expected in values.items():
            path = (folder / relative).resolve()
            require(path.is_relative_to(folder), 'Manifest path leaves its directory')
            require(self.hashed_file(path) == expected, f'Artifact hash differs: {path}')
        return values

    def inputs(self):
        p = self.protocol
        require(p['status'] == 'frozen', 'Protocol is not frozen')
        location = Path(next(iter(importlib.util.find_spec('ml441_assignment4').submodule_search_locations)))
        actual = {path.name: self.hashed_file(path) for path in sorted(location.glob('*.py'))}
        equal(actual, p['implementation_code'], 'frozen implementation')
        env = {'python': platform.python_version(), 'platform': platform.platform(), 'architecture': platform.machine(),
               'packages': {key: importlib.metadata.version(key) for key in p['execution_environment']['packages']}}
        equal(env, p['execution_environment'], 'audit environment')
        for item in p['gate_evidence']:
            require(self.hashed_file(self.root / item['path']) == item['sha256'], 'Sealed gate evidence differs')
        for name, expected in [(p['data']['file'], p['data']['sha256']), ('data/raw/features.csv', p['data']['features_sha256']),
                               ('data/raw/attack_category_map.csv', p['data']['class_map_sha256']), (p['splits']['source'], p['splits']['artifact_sha256'])]:
            require(self.hashed_file(self.root / name) == expected, 'Input provenance differs')
        raw = pd.read_csv(self.root / p['data']['file'], na_values=['?'], low_memory=False)
        require(raw.shape == (p['data']['rows'], 44), 'Raw dimensions differ')
        self.labels = np.asarray(p['data']['class_order'])
        mapping = pd.read_csv(self.root / 'data/raw/attack_category_map.csv')
        equal(mapping['Mapping'].tolist(), self.labels.tolist(), 'class mapping')
        equal(mapping['Attack Category Name'].tolist(), p['data']['class_names'], 'class names')
        self.target = pd.to_numeric(raw[p['data']['target']], errors='raise').to_numpy(dtype=np.int8)
        self.features = raw.drop(columns=[p['data']['target'], *p['representation']['drop']]).copy()
        for column in p['representation']['nominal']:
            self.features[column] = self.features[column].astype('string').fillna('Missing')
        numeric = [c for c in self.features if c not in p['representation']['nominal']]
        require(np.isfinite(self.features[numeric].to_numpy()).all(), 'Nonfinite corrected numeric value')
        groups, _ = pd.factorize(pd.util.hash_pandas_object(self.features, index=False, categorize=True), sort=False)
        distinct = self.features.assign(audit_group=groups).drop_duplicates()
        require(not distinct['audit_group'].duplicated().any(), 'Hash collision between unequal corrected vectors')
        with np.load(self.root / p['splits']['source'], allow_pickle=False) as archive:
            self.partitions = {key: archive[key] for key in archive.files}
        require(np.array_equal(groups, self.partitions['corrected_group']), 'Corrected vector groups differ')
        self.mapping = self.partitions['outer_fold']
        rows = np.arange(len(self.target))
        require(self.mapping.shape == rows.shape and np.isin(self.mapping, np.arange(1, 6)).all(), 'Invalid outer mapping')
        for fold in p['splits']['fold_index_hashes']:
            f = fold['fold']
            for role in ['outer_train', 'outer_test', 'inner_train', 'inner_validation']:
                values = self.partitions[f'fold_{f}_{role}']
                require(ordered_hash(values) == fold[role], 'Ordered split hash differs')
                require(len(np.unique(values)) == len(values) and np.isin(values, rows).all(), 'Invalid split rows')
            outer_fit, outer_test, inner_fit, inner_test = [self.partitions[f'fold_{f}_{key}'] for key in ['outer_train', 'outer_test', 'inner_train', 'inner_validation']]
            require(np.array_equal(outer_test, rows[self.mapping == f]), 'Historical outer membership differs')
            require(np.array_equal(np.sort(np.r_[outer_fit, outer_test]), rows), 'Outer rows incomplete')
            require(np.array_equal(np.sort(np.r_[inner_fit, inner_test]), outer_fit), 'Inner rows incomplete')
            for fit, test in [(outer_fit, outer_test), (inner_fit, inner_test)]:
                require(not np.intersect1d(groups[fit], groups[test]).size, 'Corrected group crosses partition')

    def representation(self, folder, fitting, evaluation):
        if folder not in self.prepared:
            require(set(self.manifest(folder, 'preprocessing_manifest.json')) == {'representation.json', 'indices.npz', 'preprocessor.joblib', 'preprocessing_cost.json'}, 'Preprocessing manifest schema differs')
            rep = read(folder / 'representation.json')
            with np.load(folder / 'indices.npz', allow_pickle=False) as saved:
                require(np.array_equal(saved['fitting'], fitting) and np.array_equal(saved['evaluation'], evaluation), 'Encoder partition differs')
            require(rep['fitting_index_sha256'] == ordered_hash(fitting), 'Encoder fitting hash differs')
            equal(rep['input_columns'], self.features.columns.tolist(), 'encoder inputs')
            equal(rep['nominal_columns'], self.protocol['representation']['nominal'], 'nominal predictors')
            numeric = [c for c in self.features if c not in rep['nominal_columns']]
            equal(rep['numeric_columns'], numeric, 'numeric predictors')
            for column, levels in zip(rep['nominal_columns'], rep['categories'], strict=True):
                equal(levels, sorted(self.features.iloc[fitting][column].unique().tolist()), 'train-only categories')
            expected_columns = numeric + [f'{c}_{level}' for c, levels in zip(rep['nominal_columns'], rep['categories'], strict=True) for level in levels]
            equal(rep['output_columns'], expected_columns, 'encoded coordinates')
            require(rep['dtype'] == 'float32', 'Matrix representation differs')
            self.prepared[folder] = rep
        return self.prepared[folder]

    def bootstrap(self, record, n):
        identity, costs = record['identity'], record['costs']
        trees, seed = identity['trees'], identity['seed']
        expected_seeds = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=trees).tolist()
        equal(costs['tree_seeds'], expected_seeds, 'member seeds')
        key = (n, seed, trees)
        if key not in self.regenerated:
            self.regenerated[key] = [ordered_hash(np.random.RandomState(s).randint(0, n, n).astype(np.int32)) for s in expected_seeds]
        equal(record['bootstrap'], {'draw_sizes': [n] * trees, 'draw_sha256': self.regenerated[key]}, 'uniform row bootstrap')

    def members(self, folder, record, evaluation, predicted):
        with np.load(folder / 'members.npz', allow_pickle=False) as saved:
            require(np.array_equal(saved['row_index'], evaluation), 'Member rows differ')
            hard = saved['hard_predictions']
        trees = record['identity']['trees']
        require(hard.shape == (len(evaluation), trees) and np.isin(hard, self.labels).all(), 'Member predictions differ')
        votes = np.stack([(hard == c).sum(axis=1) for c in self.labels], axis=1)
        pairs = trees * (trees - 1) // 2
        errors = (hard != self.target[evaluation, None]).sum(axis=1)
        disagreement = float(np.mean((trees * trees - (votes * votes).sum(axis=1)) / (trees * (trees - 1))))
        result = {'members': trees, 'pairs': pairs, 'mean_pairwise_disagreement': disagreement, 'pairwise_disagreement': None,
                  'mean_member_error': float(errors.sum() / (len(evaluation) * trees)),
                  'mean_pairwise_simultaneous_error': float(np.mean(errors * (errors - 1) / (trees * (trees - 1)))),
                  'hard_vote_disagreements': int(np.sum(votes.argmax(axis=1) != predicted))}
        equal(record['diversity'], result, 'member diagnostics')
        base = {k: record['identity'][k] for k in ['protocol_sha256', 'code', 'environment', 'role']}
        supplementary = self.root / 'output/supplementary' / canonical(base)[:16] / folder.relative_to(self.directory) / 'diversity.json'
        equal(read(supplementary), {'role': 'Predeclared inner mechanism diagnostic, no outer selection effect', 'identity': record['identity'], **result}, 'supplementary diagnostics')

    def record(self, path):
        folder = path.parent
        manifest = self.manifest(folder, 'artifacts.json')
        r = read(folder / 'record.json')
        identity = r['identity']
        stage, fold, seed, trees = [identity[key] for key in ['stage', 'fold', 'seed', 'trees']]
        require(fold in range(1, 6) and seed in self.protocol['ensemble']['seeds'], 'Invalid fold or seed')
        require(stage in ['structure', 'tree_count', 'outer'], 'Invalid record stage')
        require(identity['candidate_id'] in candidates(self.protocol), 'Unknown structural candidate')
        equal(identity['setting'], candidates(self.protocol)[identity['candidate_id']], 'structural controls')
        expected_manifest = {'record.json', 'predictions.npz', 'complete.json', 'model.joblib'} | ({'members.npz'} if stage != 'outer' else set())
        require(set(manifest) == expected_manifest, 'Run manifest schema differs')
        prefix = f'fold_{fold}_' + ('outer_' if stage == 'outer' else 'inner_')
        fitting = self.partitions[prefix + 'train']
        evaluation = self.partitions[prefix + ('test' if stage == 'outer' else 'validation')]
        prepared = folder.parent if stage == 'outer' else self.directory / 'selection' / f'fold_{fold}'
        rep = self.representation(prepared, fitting, evaluation)
        expected_identity = {'protocol_sha256': self.seal, 'code': self.protocol['implementation_code'], 'environment': self.protocol['execution_environment'],
                             'role': 'primary', 'fold': fold, 'candidate_id': identity['candidate_id'], 'setting': identity['setting'], 'seed': seed, 'trees': trees,
                             'fitting_sha256': ordered_hash(fitting), 'evaluation_sha256': ordered_hash(evaluation), 'representation_sha256': canonical(rep),
                             'class_order': self.labels.tolist(), 'stage': stage}
        equal(identity, expected_identity, 'run identity')
        if stage == 'structure':
            require(trees == 100, 'Structural tree count differs')
            expected_folder = prepared / identity['candidate_id'] / f'seed_{seed}_n100'
        elif stage == 'tree_count':
            require(trees in [50, 200], 'Unexpected count-stage checkpoint')
            expected_folder = prepared / 'tree_count' / f'seed_{seed}_n{trees}'
        else:
            expected_folder = self.directory / 'evaluation' / f'fold_{fold}' / f'seed_{seed}'
        require(folder == expected_folder, 'Record path contradicts identity')
        calculated, predicted = prediction_check(folder / 'predictions.npz', evaluation, self.target, self.labels)
        equal(r['metrics'], calculated, 'independent metric recalculation')
        complete = read(folder / 'complete.json')
        equal(complete['identity'], identity, 'checkpoint identity')
        require(complete['identity_sha256'] == canonical(identity), 'Checkpoint identity seal differs')
        require(complete['model_sha256'] == self.hashed_file(folder / 'model.joblib'), 'Checkpoint content differs')
        require(complete['model_bytes'] == r['checkpoint_bytes'] == (folder / 'model.joblib').stat().st_size, 'Checkpoint byte count differs')
        parameters = {**self.protocol['ensemble']['fixed'], **identity['setting'], 'n_estimators': trees, 'random_state': seed,
                      'n_jobs': 1, 'verbose': 0, 'warm_start': stage == 'tree_count' and trees == 200}
        equal(r['forest_parameters'], parameters, 'estimator controls')
        require(r['fit_tree_workers'] == 4 and r['prediction_tree_workers'] == 1, 'Worker controls differ')
        require(r['new_trees'] == (0 if trees == 50 and stage == 'tree_count' else 100 if stage == 'tree_count' else trees), 'New-tree budget count differs')
        require((r['fit_cost'] is None) == (stage == 'tree_count' and trees == 50), 'Derived prefix has incorrect timing scope')
        for cost in ['fit_cost', 'prediction_cost', 'diagnostic_cost', 'serialisation_cost']:
            if r[cost] is not None:
                require(np.isfinite(r[cost]['elapsed_seconds']) and r[cost]['elapsed_seconds'] >= 0, 'Invalid elapsed cost')
                require(0 < r[cost]['sampled_process_peak_rss_bytes'] <= r[cost]['limit_bytes'] and not r[cost]['limit_exceeded'], 'Memory limit exceeded')
        require(datetime.fromisoformat(r['started_utc']) <= datetime.fromisoformat(r['completed_utc']), 'Run timestamps reversed')
        require(datetime.fromisoformat(r['started_utc']) >= datetime.fromisoformat(self.protocol['frozen_at_utc']), 'Primary record preceded protocol freeze')
        require(r['costs']['trees'] == trees and len(r['costs']['max_depths']) == trees and len(r['costs']['node_feature_limits']) == trees, 'Tree cost lengths differ')
        nominal_features = max(1, int(np.sqrt(len(rep['output_columns'])))) if identity['setting']['max_features'] == 'sqrt' else len(rep['output_columns'])
        require(r['costs']['node_feature_limits'] == [nominal_features] * trees, 'Nominal node feature counts differ')
        self.bootstrap(r, len(fitting))
        if stage != 'outer':
            self.members(folder, r, evaluation, predicted)
        key = (stage, fold, identity['candidate_id'], seed, trees)
        require(key not in self.records, 'Duplicate run identity')
        self.records[key] = r

    def selections(self):
        selected = []
        evidence = {}
        for path in self.selected_paths:
            saved = read(path)
            fold = saved['fold']
            rows = [r for key, r in self.records.items() if key[0] == 'structure' and key[1] == fold]
            require(len(rows) == len(candidates(self.protocol)) * len(self.protocol['ensemble']['seeds']), 'Completed selection omits structural candidates')
            winner = choose([summary_row(r) for r in rows], self.protocol['ensemble']['seeds'], self.protocol['selection']['structural_numerical_tie_tolerance'])
            controls = candidates(self.protocol)[winner['candidate_id']]
            count_rows = []
            for seed in self.protocol['ensemble']['seeds']:
                member_arrays = {}
                for count in [50, 100, 200]:
                    stage = 'structure' if count == 100 else 'tree_count'
                    record = self.records[(stage, fold, winner['candidate_id'], seed, count)]
                    count_rows.append(summary_row(record, f'trees_{count:04d}'))
                    folder = path.parent / (winner['candidate_id'] if count == 100 else 'tree_count') / f'seed_{seed}_n{count}'
                    with np.load(folder / 'members.npz', allow_pickle=False) as archive:
                        member_arrays[count] = archive['hard_predictions']
                require(np.array_equal(member_arrays[50], member_arrays[100][:, :50]), 'Fifty-tree prefix members differ')
                require(np.array_equal(member_arrays[100], member_arrays[200][:, :100]), 'Continuation changed existing member predictions')
            count_winner = choose(count_rows, self.protocol['ensemble']['seeds'], self.protocol['selection']['structural_numerical_tie_tolerance'], count_stage=True)
            expected = {'fold': fold, 'candidate_id': winner['candidate_id'], 'setting': controls, 'trees': int(count_winner['trees']),
                        'structural_selection': winner, 'count_selection': count_winner}
            equal(saved, expected, 'reconstructed fold selection')
            selected.append(expected)
            for record_path in self.snapshot:
                if record_path.is_relative_to(path.parent):
                    evidence[str(record_path.relative_to(self.directory))] = self.hashed_file(record_path)
            evidence[str(path.relative_to(self.directory))] = self.hashed_file(path)
        lock_path = self.directory / 'selection_lock.json'
        if self.lock_at_start:
            require(len(selected) == 5, 'Selection lock without five reconstructed selections')
            lock = read(lock_path)
            self.hashed_file(lock_path)
            equal(lock['selections'], selected, 'locked selections')
            equal(lock['evidence'], evidence, 'complete selection evidence')
            equal(lock['identity'], {'protocol_sha256': self.seal, 'code': self.protocol['implementation_code'], 'environment': self.protocol['execution_environment'], 'role': 'primary'}, 'selection-lock identity')
            for key, record in self.records.items():
                if key[0] == 'outer':
                    require(datetime.fromisoformat(record['started_utc']) >= datetime.fromisoformat(lock['locked_utc']), 'Outer fit preceded selection lock')
                    chosen = next(s for s in selected if s['fold'] == key[1])
                    require(key[2] == chosen['candidate_id'] and key[4] == chosen['trees'], 'Outer controls differ from selection')
        else:
            require(not any(key[0] == 'outer' for key in self.records), 'Outer evidence before the selection lock')
        matched = {}
        for key, record in self.records.items():
            if key[0] == 'structure':
                setting = record['identity']['setting']
                pair = (key[1], key[3], setting['min_samples_leaf'], setting['class_weight'])
                if pair in matched:
                    equal(record['bootstrap'], matched[pair], 'matched feature-sampling bootstrap draws')
                matched[pair] = record['bootstrap']
        return selected

    def complete_summary(self):
        p = self.protocol
        history = self.root / 'materials/assignment2_baseline_records/outer_predictions.csv.gz'
        require(self.hashed_file(history) == p['comparators']['prediction_sha256'], 'Comparator prediction provenance differs')
        data = pd.read_csv(history)
        require(sorted(data.model.unique()) == ['Classification tree', 'kNN'], 'Both historical comparators are required')
        baselines = {}
        for model, values in data.groupby('model', sort=True):
            values = values.sort_values('row_index')
            require(np.array_equal(values.row_index, np.arange(len(self.target))) and np.array_equal(values.true, self.target), 'Historical row or target mismatch')
            require(np.array_equal(values.outer_fold, self.mapping), 'Historical fold mismatch')
            folds = {str(f): recalculate(rows.true.to_numpy(), rows.predicted.to_numpy(), self.labels) for f, rows in values.groupby('outer_fold')}
            baselines[model] = {'folds': folds, 'pooled': recalculate(self.target, values.predicted.to_numpy(), self.labels)}
        seed_rows, pooled = [], {}
        for seed in p['ensemble']['seeds']:
            predictions = np.empty(len(self.target), dtype=np.int8)
            coverage = np.zeros(len(self.target), dtype=np.int8)
            for fold in range(1, 6):
                runs = [r for key, r in self.records.items() if key[0] == 'outer' and key[1] == fold and key[3] == seed]
                require(len(runs) == 1, 'Incomplete or duplicate outer fold-seed record')
                folder = self.directory / 'evaluation' / f'fold_{fold}' / f'seed_{seed}'
                with np.load(folder / 'predictions.npz', allow_pickle=False) as archive:
                    predictions[archive['row_index']] = archive['predicted']
                    coverage[archive['row_index']] += 1
                seed_rows.append({'fold': fold, 'seed': seed, **runs[0]['metrics']})
            require(np.all(coverage == 1), 'Pooled rows are missing or repeated')
            pooled[str(seed)] = recalculate(self.target, predictions, self.labels)
        folds = []
        for fold in range(1, 6):
            scores = [r['macro_f1'] for r in seed_rows if r['fold'] == fold]
            folds.append({'fold': fold, 'macro_f1_seed_mean': float(np.mean(scores)), 'macro_f1_seed_sd': float(np.std(scores, ddof=1))})
        paired = {name: [{'fold': row['fold'], 'ensemble_seed_mean_macro_f1': row['macro_f1_seed_mean'],
                          'comparator_macro_f1': baseline['folds'][str(row['fold'])]['macro_f1'],
                          'difference': row['macro_f1_seed_mean'] - baseline['folds'][str(row['fold'])]['macro_f1']} for row in folds] for name, baseline in baselines.items()}
        expected = {'status': 'verified', 'evidence_role': p['evidence_role'], 'folds': folds, 'seed_fold_records': seed_rows,
                    'macro_f1_fold_mean': float(np.mean([r['macro_f1_seed_mean'] for r in folds])),
                    'macro_f1_fold_sd': float(np.std([r['macro_f1_seed_mean'] for r in folds], ddof=1)),
                    'pooled_by_seed': pooled, 'baselines': baselines, 'paired_comparisons': paired,
                    'inference': 'Descriptive paired folds; no independence or formal significance claim'}
        equal(read(self.directory / 'comparison_summary.json'), expected, 'independent comparison summary')
        outer = read(self.directory / 'outer_records.json')
        indexed = {(r['fold'], r['seed']): r for r in outer}
        require(len(outer) == len(indexed) == 15, 'Outer summary coverage differs')
        for key, record in self.records.items():
            if key[0] == 'outer':
                equal(indexed[(key[1], key[3])], {'fold': key[1], 'seed': key[3], **record['metrics'], 'checkpoint_bytes': record['checkpoint_bytes']}, 'outer summary record')
        return {'comparison_summary_sha256': self.hashed_file(self.directory / 'comparison_summary.json'), 'independent_summary': expected}

    def run(self):
        self.inputs()
        for i, path in enumerate(self.snapshot, 1):
            self.record(path)
            if i % 20 == 0:
                print(json.dumps({'event': 'completed_records_verified', 'count': i, 'snapshot_total': len(self.snapshot)}), flush=True)
        selected = self.selections()
        counts = {stage: sum(key[0] == stage for key in self.records) for stage in ['structure', 'tree_count', 'outer']}
        fits = sum(r['fit_cost'] is not None for r in self.records.values())
        new_trees = sum(r['new_trees'] for r in self.records.values())
        seconds = sum(r['fit_cost']['elapsed_seconds'] for r in self.records.values() if r['fit_cost'] is not None)
        require(fits <= self.protocol['budget']['total_fit_or_continuation_calls'] and new_trees <= self.protocol['budget']['maximum_new_primary_trees'], 'Primary budget counts exceeded')
        require(seconds < self.protocol['budget']['maximum_fit_hours'] * 3600, 'Primary fit-time budget exceeded')
        extra = {}
        if self.mode == 'complete':
            require(counts == {'structure': 180, 'tree_count': 30, 'outer': 15}, 'Complete audit requires all 225 saved estimator/prefix records')
            extra = self.complete_summary()
        return {'status': 'complete_saved_record_audit_passed' if self.mode == 'complete' else 'completed_record_snapshot_passed',
                'mode': self.mode, 'started_utc': self.started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': self.seal, 'auditor_sha256': digest(__file__), 'record_counts': counts,
                'snapshot_artifacts': [str(path.relative_to(self.root)) for path in self.snapshot],
                'verified_files_sha256': {str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path): value for path, value in self.hashed.items()},
                'reconstructed_fold_selections': len(selected), 'fit_or_continuation_calls': fits, 'newly_grown_trees': new_trees,
                'fit_elapsed_seconds': seconds, 'primary_outputs_modified': False, 'estimator_fits_performed': 0,
                'primary_comparison_ready_for_report': self.mode == 'complete',
                'independence': 'Separate implementation; no primary package metric, selection, verification or summary functions imported',
                'limitations': ['Snapshot mode covers only immutable completed manifests captured at audit start',
                               'Artifact hashes and saved probabilities are audited; every checkpoint is not refitted or replayed',
                               'Encoder fitting categories are independently reconstructed; transformations are not replayed for every record',
                               'Historical data reuse and dependent outer folds remain limitations',
                               'Auditing adds host load; elapsed measurements are descriptive rather than controlled hardware benchmarks'], **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['snapshot', 'complete'], required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.project_root / 'output/verification/primary_audits' / stamp
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = Audit(args.project_root, args.mode).run()
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'status': 'failed', 'mode': args.mode, 'type': type(error).__name__, 'message': str(error)}, indent=2) + '\n')
        raise
    (output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'record_counts': result['record_counts'], 'comparison_ready': result['primary_comparison_ready_for_report'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
