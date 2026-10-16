import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import config_context
from threadpoolctl import threadpool_limits

from audit_primary_records import canonical, digest, equal, ordered_hash, recalculate, require
from ml441_assignment4.baselines import fit_historical_neighbours, fit_historical_tree, historical_preprocessor, neighbour_vote
from ml441_assignment4.data import corrected_groups, load_data
from ml441_assignment4.models import make_forest
from ml441_assignment4.preprocessing import forest_matrix, make_preprocessor, representation_record
from ml441_assignment4.primary import validate_frozen_protocol
from ml441_assignment4.records import MemoryMonitor, save_checkpoint, write_json
from ml441_assignment4.splits import load_verified_splits


def read(path):
    return json.loads(Path(path).read_text())


def preflight(root):
    spec = read(root / 'configs/comparable_cost_protocol.json')
    require(canonical(spec) == (root / 'configs/comparable_cost_protocol.sha256').read_text().strip(), 'Cost protocol seal differs')
    for relative, expected in spec['implementation'].items():
        require(digest(root / relative) == expected, 'Cost implementation differs from sealed registration')
    primary = read(root / 'configs/primary_protocol.json')
    validate_frozen_protocol(root, primary)
    require(canonical(primary) == spec['primary_protocol_sha256'], 'Primary protocol differs')
    directory = root / 'output/primary' / canonical(primary)[:16]
    require(not (directory / 'process.lock').exists(), 'Cost study waits for the primary process to finish')
    launch = read(root / 'output/primary/launch.json')
    try:
        os.kill(launch['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Recorded primary process is still alive')
    require(not (directory / 'failure.json').exists(), 'Primary failure must be resolved before cost reproduction')
    audits = sorted((root / 'output/verification/primary_audits').glob('*/audit.json'))
    valid = [(path, read(path)) for path in audits if read(path).get('status') == 'complete_saved_record_audit_passed']
    require(bool(valid), 'A complete independent saved-record audit is required')
    audit_path, audit = valid[-1]
    require(audit['protocol_sha256'] == canonical(primary) and audit['auditor_sha256'] == digest(root / 'src/audit_primary_records.py'), 'Complete audit identity differs')
    require(audit['record_counts'] == {'structure': 180, 'tree_count': 30, 'outer': 15} and audit['primary_comparison_ready_for_report'], 'Complete audit coverage is insufficient')
    for name in ['selection_lock.json', 'comparison_summary.json']:
        path = directory / name
        require(digest(path) == audit['verified_files_sha256'][str(path.relative_to(root))], 'Audited primary summary or lock has changed')
    for fold in range(1, 6):
        for seed in spec['forest_seeds']:
            path = directory / 'evaluation' / f'fold_{fold}' / f'seed_{seed}' / 'predictions.npz'
            require(digest(path) == audit['verified_files_sha256'][str(path.relative_to(root))], 'Audited reference prediction has changed')
    for name, expected in [('materials/assignment2_baseline_records/selected_parameters.json', spec['inputs']['baseline_settings_sha256']),
                           ('materials/assignment2_baseline_records/outer_predictions.csv.gz', spec['inputs']['baseline_predictions_sha256'])]:
        require(digest(root / name) == expected, 'Historical comparator input differs')
    require(spec['environment'] == primary['execution_environment'] and spec['immutable_primary_code'] == primary['implementation_code'], 'Cost environment or shared code differs')
    return spec, primary, directory, audit_path


def fit_model(model, spec, protocol, setting, seed, fold, matrix, target):
    workers = spec['workers']
    if model == 'forest':
        estimator = make_forest(protocol, setting['setting'], seed, setting['trees'], workers=workers['forest_fit'])
        estimator.fit(matrix, target)
        estimator.set_params(n_jobs=workers['forest_prediction'])
        return {'estimator': estimator}
    if model == 'classification_tree':
        return {'estimator': fit_historical_tree(matrix, target, setting, fold)}
    require(model == 'knn', 'Unknown cost-study model')
    estimator, labels, positions = fit_historical_neighbours(matrix, target, setting, fold, protocol['data']['class_order'])
    estimator.set_params(n_jobs=workers['nearest_neighbour_search'])
    return {'estimator': estimator, 'prototype_labels': labels, 'prototype_positions': positions}


def predict_model(model, payload, setting, matrix, labels):
    if model == 'knn':
        distances, indices = payload['estimator'].kneighbors(matrix, return_distance=True)
        return neighbour_vote(distances, indices, payload['prototype_labels'], labels, setting)
    if model == 'forest':
        probability = payload['estimator'].predict_proba(matrix)
        require(np.array_equal(payload['estimator'].classes_, labels), 'Forest class alignment differs')
        return np.asarray(labels)[probability.argmax(axis=1)]
    return payload['estimator'].predict(matrix)


def verify_completed(folder, identity, evaluation, target, expected):
    path = folder / 'artifacts.json'
    if not path.exists():
        return None
    manifest = read(path)
    require(set(manifest) == {'record.json', 'predictions.npz', 'complete.json', 'model.joblib', 'preprocessor.joblib', 'representation.json', 'indices.npz'}, 'Cost artifact manifest differs')
    for name, value in manifest.items():
        require(digest(folder / name) == value, 'Completed cost artifact has changed')
    record = read(folder / 'record.json')
    require(record['identity'] == identity, 'Completed cost identity differs')
    complete = read(folder / 'complete.json')
    require(complete['identity'] == identity and complete['identity_sha256'] == canonical(identity), 'Cost checkpoint identity differs')
    require(complete['model_sha256'] == manifest['model.joblib'], 'Cost model seal differs')
    require(complete['model_bytes'] == record['model_checkpoint_bytes'] == (folder / 'model.joblib').stat().st_size, 'Cost checkpoint size differs')
    require(manifest['preprocessor.joblib'] == record['encoder_sha256'], 'Cost encoder seal differs')
    require(canonical(read(folder / 'representation.json')) == record['representation_sha256'], 'Cost representation differs')
    with np.load(folder / 'indices.npz', allow_pickle=False) as indices:
        require(ordered_hash(indices['fitting']) == identity['fitting_sha256'] and np.array_equal(indices['evaluation'], evaluation), 'Cost fitting or evaluation rows differ')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
        require(np.array_equal(saved['row_index'], evaluation) and np.array_equal(saved['true'], target[evaluation]), 'Saved cost prediction rows differ')
        require(np.array_equal(saved['expected'], expected), 'Saved comparison predictions differ')
        require(int(np.sum(saved['predicted'] != expected)) == record['prediction_disagreements'], 'Saved agreement count differs')
        equal(record['recalculated_metrics'], recalculate(target[evaluation], saved['predicted'], record['recalculated_metrics']['labels']), 'resumed cost metrics')
    return record


def run_one(root, output, model, spec, protocol, directory, audit_path, features, target, partitions, fold, seed, setting, history):
    fitting, evaluation = [partitions[f'fold_{fold}_outer_{name}'] for name in ['train', 'test']]
    folder = output / f'fold_{fold}' / model / f'seed_{seed}'
    identity = {'cost_protocol_sha256': canonical(spec), 'primary_protocol_sha256': canonical(protocol),
                'complete_audit_sha256': digest(audit_path), 'fold': fold, 'model': model, 'seed': seed, 'setting': setting,
                'fitting_sha256': ordered_hash(fitting), 'evaluation_sha256': ordered_hash(evaluation)}
    if model == 'forest':
        with np.load(directory / 'evaluation' / f'fold_{fold}' / f'seed_{seed}' / 'predictions.npz', allow_pickle=False) as saved:
            require(np.array_equal(saved['row_index'], evaluation), 'Primary reference rows differ')
            expected = saved['predicted']
    else:
        name = 'Classification tree' if model == 'classification_tree' else 'kNN'
        rows = history[(history.model == name) & (history.outer_fold == fold)].sort_values('row_index')
        require(np.array_equal(rows.row_index, evaluation), 'Historical reference rows differ')
        expected = rows.predicted.to_numpy()
    prior = verify_completed(folder, identity, evaluation, target, expected)
    if prior is not None:
        return prior
    if folder.exists() and any(folder.iterdir()):
        archive = output / 'incomplete' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + f'fold_{fold}_{model}_seed_{seed}')
        archive.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(archive)
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    write_json(folder / 'started.json', {'identity': identity, 'started_utc': started, 'pid': os.getpid()})
    with config_context(working_memory=spec['workers']['working_memory_mib']), threadpool_limits(limits=spec['workers']['native_threadpool_limit']):
        with MemoryMonitor(limit_gib=spec['budget']['process_memory_gib']) as preprocess_cost:
            encoder = make_preprocessor(features.iloc[fitting], protocol['representation']['nominal']) if model == 'forest' else historical_preprocessor(features.iloc[fitting], protocol['representation']['nominal'], neighbour=model == 'knn')
            fit_matrix = forest_matrix(encoder.fit_transform(features.iloc[fitting]), fitting=model != 'knn')
            evaluation_matrix = forest_matrix(encoder.transform(features.iloc[evaluation]))
        with MemoryMonitor(limit_gib=spec['budget']['process_memory_gib']) as fit_cost:
            payload = fit_model(model, spec, protocol, setting, seed, fold, fit_matrix, target[fitting])
        with MemoryMonitor(limit_gib=spec['budget']['process_memory_gib']) as prediction_cost:
            predicted = predict_model(model, payload, setting, evaluation_matrix, protocol['data']['class_order'])
    differences = int(np.sum(predicted != expected))
    representation = representation_record(encoder)
    representation['fitting_index_sha256'] = ordered_hash(fitting)
    import joblib
    with MemoryMonitor(limit_gib=spec['budget']['process_memory_gib']) as serialisation_cost:
        model_path = save_checkpoint(folder, identity, payload)
        joblib.dump(encoder, folder / 'preprocessor.joblib', compress=0)
        write_json(folder / 'representation.json', representation)
        np.savez_compressed(folder / 'indices.npz', fitting=fitting, evaluation=evaluation)
    np.savez_compressed(folder / 'predictions.npz', row_index=evaluation, true=target[evaluation], predicted=predicted, expected=expected)
    record = {'identity': identity, 'status': 'exact_prediction_reproduction' if differences == 0 else 'reproduction_difference',
              'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'preprocessing_cost': preprocess_cost.record(), 'fit_cost': fit_cost.record(), 'prediction_cost': prediction_cost.record(),
              'serialisation_cost': serialisation_cost.record(), 'model_checkpoint_bytes': model_path.stat().st_size,
              'encoder_bytes': (folder / 'preprocessor.joblib').stat().st_size,
              'encoder_sha256': digest(folder / 'preprocessor.joblib'), 'representation_sha256': canonical(representation),
              'prediction_disagreements': differences, 'evaluation_rows': len(evaluation),
              'recalculated_metrics': recalculate(target[evaluation], predicted, protocol['data']['class_order']),
              'run_wall_seconds': time.monotonic() - wall_start,
              'timing_boundaries': spec['timing_boundaries'], 'workers': spec['workers'],
              'storage_scope': 'Estimator plus identity wrapper; encoder separate; neighbour prototype labels and positions included; diagnostic arrays separate'}
    if model == 'knn':
        record['prototype_row_index_sha256'] = ordered_hash(fitting[payload['prototype_positions']])
        record['prototype_rows'] = len(payload['prototype_positions'])
    write_json(folder / 'record.json', record)
    write_json(folder / 'artifacts.json', {name: digest(folder / name) for name in ['record.json', 'predictions.npz', 'complete.json', 'model.joblib', 'preprocessor.joblib', 'representation.json', 'indices.npz']})
    del payload, fit_matrix, evaluation_matrix, encoder
    gc.collect()
    return record


def run(root):
    spec, protocol, primary_directory, audit_path = preflight(root)
    output = root / spec['output_root'] / canonical(spec)[:16]
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'process.lock'
    if lock.exists():
        previous = int(lock.read_text())
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        start = time.monotonic()
        features, target = load_data(root, protocol)
        partitions = load_verified_splits(root, protocol, corrected_groups(features))
        load_seconds = time.monotonic() - start
        history = pd.read_csv(root / 'materials/assignment2_baseline_records/outer_predictions.csv.gz')
        baseline = {r['outer_fold']: r for r in read(root / 'materials/assignment2_baseline_records/selected_parameters.json')}
        selected = {r['fold']: r for r in read(primary_directory / 'selection_lock.json')['selections']}
        records = []
        write_json(output / 'protocol.json', spec)
        for fold in spec['fold_order']:
            for model in spec['pipeline_order']:
                seeds = spec['forest_seeds'] if model == 'forest' else [20260826 + fold]
                setting = selected[fold] if model == 'forest' else baseline[fold]['classification_tree' if model == 'classification_tree' else 'knn']
                for seed in seeds:
                    require(sum(r['run_wall_seconds'] for r in records) + load_seconds < spec['budget']['maximum_elapsed_hours'] * 3600, 'Cost-study elapsed budget reached between models')
                    stored = sum(p.stat().st_size for p in output.rglob('*.joblib'))
                    require(stored < spec['budget']['checkpoint_storage_gib'] * 1024 ** 3, 'Cost-study checkpoint budget reached')
                    print(json.dumps({'event': 'cost_model_start', 'fold': fold, 'model': model, 'seed': seed}), flush=True)
                    record = run_one(root, output, model, spec, protocol, primary_directory, audit_path, features, target, partitions, fold, seed, setting, history)
                    records.append(record)
                    write_json(output / 'progress.json', {'completed_models': len(records), 'last_identity': record['identity'], 'last_status': record['status']})
            if fold == 1:
                projection = 5 * sum(r['run_wall_seconds'] for r in records) + load_seconds
                write_json(output / 'first_fold_cost_review.json', {'projected_total_seconds': projection, 'budget_seconds': spec['budget']['maximum_elapsed_hours'] * 3600,
                           'scope': 'Five-fold engineering projection from fixed fold one; not an uncertainty interval', 'uses_selection_scores': False})
                require(projection <= spec['budget']['maximum_elapsed_hours'] * 3600, 'First-fold cost projection exceeds the registered budget')
        require(len(records) == spec['budget']['maximum_total_models'], 'Incomplete cost-study model coverage')
        write_json(output / 'summary.json', {'status': 'complete_exact_reproduction' if all(r['prediction_disagreements'] == 0 for r in records) else 'complete_with_reproduction_differences',
                   'cost_protocol_sha256': canonical(spec), 'complete_audit_sha256': digest(audit_path), 'raw_loading_seconds': load_seconds,
                   'records': records, 'limitations': spec['limitations'], 'primary_scores_replaced': False})
        (output / 'failure.json').unlink(missing_ok=True)
    except Exception as error:
        write_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['check', 'run'], default='check')
    args = parser.parse_args()
    if args.mode == 'check':
        try:
            preflight(args.project_root.resolve())
        except ValueError as error:
            print(json.dumps({'status': 'not_ready', 'reason': str(error), 'fits_started': 0}))
            return
        print(json.dumps({'status': 'ready', 'fits_started': 0}))
    else:
        run(args.project_root.resolve())


if __name__ == '__main__':
    main()
