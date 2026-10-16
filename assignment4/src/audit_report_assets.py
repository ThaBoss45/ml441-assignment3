import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=0, atol=2e-12), message)


def assess(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    require(matrix.shape == (10, 10) and np.all(matrix >= 0), 'Invalid fixed-label confusion')
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(10), where=support + predicted != 0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'support': support,
            'macro_f1': f1.mean(), 'balanced_accuracy': recall.mean(),
            'accuracy': correct.sum() / matrix.sum(), 'normal_false_alarm_rate': 1 - recall[0]}


def nested(values):
    require(set(values) == set(range(1, 6)), 'Five folds required')
    folded = [float(np.mean(values[fold])) for fold in range(1, 6)]
    return {'mean': float(np.mean(folded)), 'fold_sd': float(np.std(folded, ddof=1)),
            'minimum': min(folded), 'maximum': max(folded)}


def table_row(path, values):
    require(' & '.join(str(value) for value in values) + '\\\\' in path.read_text(), 'Displayed row differs: ' + path.name)


def main():
    manifest_path = ROOT / 'report/generated/asset_manifest.json'
    manifest = read(manifest_path)
    archive = ROOT / 'output/report_assets' / manifest['asset_identity']
    for path, expected in manifest['input_hashes'].items():
        require(sha(ROOT / path) == expected, 'Asset input changed: ' + path)
    for name, expected in manifest['files'].items():
        require(sha(archive / name) == expected, 'Archived asset changed: ' + name)
        if name.endswith(('.tex', '.png')):
            require(sha(ROOT / 'report/generated' / name) == expected, 'Current asset differs: ' + name)
    require(sha(ROOT / 'src/make_report_assets.py') == manifest['script_sha256'], 'Asset builder changed')
    analysis = read(archive / 'analysis.json')
    summary = read(ROOT / 'output/primary/82b87926ac5a04b6/comparison_summary.json')
    protocol = read(ROOT / 'configs/primary_protocol.json')
    tables = ROOT / 'report/generated'
    seeds = protocol['ensemble']['seeds']
    outer = summary['seed_fold_records']
    require(len(outer) == 15 and {(r['fold'], r['seed']) for r in outer} == {(f, s) for f in range(1, 6) for s in seeds}, 'Outer coverage differs')
    models = {'Ensemble': outer, 'Single tree': list(summary['baselines']['Classification tree']['folds'].values()), 'Neighbours': list(summary['baselines']['kNN']['folds'].values())}
    for model, records in models.items():
        checked = [assess(record['confusion']) for record in records]
        for original, calculated in zip(records, checked):
            for field in calculated:
                close(original[field], calculated[field], 'Source metric differs: ' + field)
        for field in ['macro_f1', 'balanced_accuracy', 'accuracy', 'normal_false_alarm_rate']:
            values = {fold: [] for fold in range(1, 6)}
            for pos, calculated in enumerate(checked):
                fold = records[pos]['fold'] if model == 'Ensemble' else pos + 1
                values[fold].append(calculated[field])
            for name, value in nested(values).items():
                close(analysis['aggregate'][model][field][name], value, 'Nested fold summary differs')
        a = analysis['aggregate'][model]
        table_row(tables / 'aggregate.tex', [model, f"{a['macro_f1']['mean']:.4f}", f"{a['macro_f1']['fold_sd']:.4f}", f"{a['balanced_accuracy']['mean']:.4f}", f"{a['normal_false_alarm_rate']['mean']:.4f}"])
    pooled = {}
    for seed in seeds:
        records = [r for r in outer if r['seed'] == seed]
        metrics = assess(np.sum([r['confusion'] for r in records], axis=0))
        pooled[str(seed)] = metrics
        for field, value in metrics.items():
            close(summary['pooled_by_seed'][str(seed)][field], value, 'Per-seed pooling differs')
        matching = next(r for r in analysis['pooled'] if r['pipeline'] == 'Ensemble ' + str(seed))
        for field, value in metrics.items():
            close(matching[field], value, 'Reported pooled metric differs')
    support = pooled[str(seeds[0])]['support']
    require(int(support.sum()) == analysis['class_support_total'] == 257673, 'Support was multiplied or changed')
    require(int(support[7]) == 174, 'Scarce-class support differs')
    for position, row in enumerate(analysis['classes']):
        require(row['support'] == support[position], 'Class support differs')
        for field in ['precision', 'recall', 'f1']:
            close(row['ensemble_' + field], np.mean([pooled[str(seed)][field][position] for seed in seeds]), 'Seed-mean class metric differs')
            for key, short in [('kNN', 'neighbours'), ('Classification tree', 'tree')]:
                close(row[short + '_' + field], assess(summary['baselines'][key]['pooled']['confusion'])[field][position], 'Baseline class metric differs')
        table_row(tables / 'classes.tex', [row['class'], row['support']] + [f"{row[model + '_' + field]:.4f}" for model in ['neighbours', 'tree', 'ensemble'] for field in ['precision', 'recall', 'f1']])
    for fold in range(1, 6):
        ensemble = [assess(r['confusion'])['macro_f1'] for r in outer if r['fold'] == fold]
        differences = []
        for model in ['kNN', 'Classification tree']:
            baseline = assess(summary['baselines'][model]['folds'][str(fold)]['confusion'])['macro_f1']
            difference = float(np.mean(ensemble) - baseline)
            differences.append(difference)
            close(summary['paired_comparisons'][model][fold - 1]['difference'], difference, 'Pairing differs')
        table_row(tables / 'paired.tex', [fold] + [f'+{d:.4f}' for d in differences] + [f'{np.std(ensemble, ddof=1):.4f}'])
    structural = [read(ROOT / path) for path in manifest['input_hashes'] if '/selection/' in path and '/tree_count/' not in path]
    require(len(structural) == 180, 'Structural coverage differs')
    for treatment in analysis['structural']:
        runs = [r for r in structural if r['identity']['setting']['min_samples_leaf'] == treatment['leaf'] and ('sqrt' if r['identity']['setting']['max_features'] == 'sqrt' else 'all') == treatment['features'] and ('None' if r['identity']['setting']['class_weight'] is None else 'Balanced') == treatment['weight']]
        require(len(runs) == 15, 'Treatment requires five folds and three seeds')
        for field in ['macro_f1', 'balanced_accuracy', 'normal_false_alarm_rate']:
            values = {fold: [assess(r['confusion']) [field] if 'confusion' in r else assess(r['metrics']['confusion'])[field] for r in runs if r['identity']['fold'] == fold] for fold in range(1, 6)}
            close(treatment[field], nested(values)['mean'], 'Treatment mean differs')
            if field in ['macro_f1', 'balanced_accuracy']:
                close(treatment[field + '_fold_sd'], nested(values)['fold_sd'], 'Treatment fold deviation differs')
        for field in ['mean_pairwise_disagreement', 'mean_member_error', 'mean_pairwise_simultaneous_error']:
            close(treatment[field], np.mean([r['diversity'][field] for r in runs]), 'Member diagnostic differs')
        table_row(tables / 'structural.tex', [int(treatment['leaf']), 'All' if treatment['features'] == 'all' else '13', treatment['weight'], f"{treatment['macro_f1']:.4f}", f"{treatment['macro_f1_fold_sd']:.4f}", f"{treatment['balanced_accuracy']:.4f}"])
    for fold in range(1, 6):
        count_values = []
        for trees in [50, 100, 200]:
            records = [read(ROOT / path) for path in manifest['input_hashes'] if f'/selection/fold_{fold}/' in path and f'_n{trees}/record.json' in path]
            if trees == 100:
                candidate = analysis['selected'][fold - 1]['candidate_id']
                records = [r for r in records if r['identity']['candidate_id'] == candidate]
            require(len(records) == 3, 'Count study seed coverage differs')
            values = [assess(r['metrics']['confusion'])['macro_f1'] for r in records]
            report = next(r for r in analysis['tree_count'] if r['fold'] == fold and r['trees'] == trees)
            close(report['mean'], np.mean(values), 'Count mean differs')
            close([report['minimum'], report['maximum'], report['seed_sd']], [min(values), max(values), np.std(values, ddof=1)], 'Count spread differs')
            require(trees != 50 or report['fit_seconds'] is None, 'Prefix time was invented')
            count_values.append(f'{np.mean(values):.4f} $\\pm$ {np.std(values, ddof=1):.4f}')
        table_row(tables / 'counts.tex', [fold] + count_values)
        selected = analysis['selected'][fold - 1]
        table_row(tables / 'selected.tex', [fold, selected['setting']['min_samples_leaf'], selected['trees'], f"{selected['count_selection']['macro_f1']:.4f}"])
    cost_path = next(path for path in manifest['input_hashes'] if '/D02/' in path and path.endswith('/summary.json'))
    costs = read(ROOT / cost_path)['records']
    require(len(costs) == 25 and all(r['prediction_disagreements'] == 0 for r in costs), 'Cost coverage/agreement differs')
    for model, title in [('knn', 'Neighbours'), ('classification_tree', 'Single tree'), ('forest', 'Ensemble')]:
        values = {field: {fold: [] for fold in range(1, 6)} for field in ['preprocessing', 'fit', 'prediction', 'serialisation', 'checkpoint_mib', 'encoder_kib', 'peak_gib']}
        for record in costs:
            if record['identity']['model'] != model:
                continue
            fold = record['identity']['fold']
            for field in ['preprocessing', 'fit', 'prediction', 'serialisation']:
                values[field][fold].append(record[field + '_cost']['elapsed_seconds'])
            values['checkpoint_mib'][fold].append(record['model_checkpoint_bytes'] / 1024 ** 2)
            values['encoder_kib'][fold].append(record['encoder_bytes'] / 1024)
            values['peak_gib'][fold].append(max(record[k + '_cost']['sampled_process_peak_rss_bytes'] for k in ['preprocessing', 'fit', 'prediction', 'serialisation']) / 1024 ** 3)
        for field, folded in values.items():
            for key, value in nested(folded).items():
                close(analysis['costs'][model][field][key], value, 'Cost summary differs')
        for filename, fields in [('costs', ['preprocessing', 'fit', 'prediction', 'serialisation']), ('storage', ['checkpoint_mib', 'encoder_kib', 'peak_gib'])]:
            table_row(tables / (filename + '.tex'), [title] + [f"{nested(values[field])['mean']:.3f} $\\pm$ {nested(values[field])['fold_sd']:.3f}" for field in fields])
    historical = read(ROOT / 'materials/assignment2_baseline_records/selected_parameters.json')
    require(historical == analysis['historical_selected'], 'Reported historical selections differ')
    require(len(historical) == 5 and [r['outer_fold'] for r in historical] == list(range(1, 6)), 'Historical coverage differs')
    for row in historical:
        for model in ['knn', 'classification_tree']:
            matches = [r['identity']['setting'] for r in costs if r['identity']['model'] == model and r['identity']['fold'] == row['outer_fold']]
            require(matches == [row[model]], 'Historical controls differ from exact predictive reproduction')
        neighbour, tree = row['knn'], row['classification_tree']
        require(neighbour['p'] == 1, 'Displayed neighbour distance differs')
        vote = 'Uniform' if neighbour['weights'] == 'uniform' else 'Distance'
        exponent = '--' if neighbour['weights'] == 'uniform' else f"{neighbour['vote_exponent']:g}"
        require(neighbour['weights'] in ['uniform', 'distance'] and (neighbour['weights'] == 'uniform' or neighbour['vote_exponent'] > 0), 'Invalid vote rule')
        table_row(tables / 'historical_neighbours.tex', [row['outer_fold'], neighbour['k'], 'Manhattan', vote, exponent])
        penalty_exponent = round(math.log10(tree['ccp_alpha']))
        close(tree['ccp_alpha'], 10. ** penalty_exponent, 'Displayed pruning penalty differs')
        table_row(tables / 'historical_tree.tex', [row['outer_fold'], tree['criterion'].capitalize(), tree['max_depth'], tree['min_samples_leaf'], 'None' if tree['class_weight'] is None else 'Balanced', '$10^{' + str(penalty_exponent) + '}$'])
    for row in analysis['pooled']:
        table_row(tables / 'pooled.tex', [row['pipeline']] + [f'{row[field]:.4f}' for field in ['macro_f1', 'balanced_accuracy', 'accuracy', 'normal_false_alarm_rate']])
    result = {'status': 'independent_report_numeric_audit_passed', 'asset_identity': manifest['asset_identity'], 'asset_manifest_sha256': sha(manifest_path),
              'audit_source_sha256': sha(__file__), 'inputs_hashed': len(manifest['input_hashes']), 'tables_checked': 11,
              'outer_folds': 5, 'outer_seed_models': 15, 'structural_records': 180, 'cost_records': 25,
              'support': int(support.sum()), 'new_fits': 0, 'zero_division': 0,
              'checks': ['Metrics independently derived from audited confusion counts', 'Five-fold summaries and sample deviations after seed averaging', 'Three-seed count sample deviations separately scoped', 'Historical selected controls matched to audited exact-prediction cost refits', 'Inactive uniform-voting exponents displayed as dashes', 'Separately pooled seed metrics', 'Paired fold differences and seed deviations', 'Class score mean never recomputed from mean precision/recall', 'Original support retained', 'All eleven formatted numeric tables matched', 'Prefix time remains null', 'Byte units and sampled-process peaks checked'],
              'scope': 'Independent report arithmetic and source integrity; primary/cost audits remain the provenance authorities; visual/language review and human proofread are separate'}
    output = ROOT / 'output/verification/report_assets_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
