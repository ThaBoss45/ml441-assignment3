import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from fontTools.agl import toUnicode
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.t1Lib import T1Font
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'output/primary/82b87926ac5a04b6'
GENERATED = ROOT / 'report/generated'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fold_summary(frame, columns):
    folded = frame.groupby('fold')[columns].mean()
    return {column: {'mean': float(folded[column].mean()), 'fold_sd': float(folded[column].std(ddof=1)),
                     'minimum': float(folded[column].min()), 'maximum': float(folded[column].max())} for column in columns}


def confusion_annotations(matrix):
    labels = []
    for row in range(len(matrix)):
        off_diagonal = np.array(matrix[row], copy=True)
        off_diagonal[row] = -1
        largest = int(np.argmax(off_diagonal))
        columns = [row] + ([largest] if off_diagonal[largest] > 0 else [])
        for column in columns:
            value = float(matrix[row, column])
            text = '<1' if 0 < value < .005 else f'{100 * value:.0f}'
            labels.append({'true': row, 'predicted': column, 'proportion': value, 'text': text})
    return labels


def table(path, caption, label, header, rows, columns=None, wide=False, prefix=''):
    environment = 'widetable' if wide else 'inlinetable'
    start = '\\widetablestart' if wide else '\\begin{inlinetable}'
    columns = columns or ('l' + 'r' * (len(header) - 1))
    lines = [start, '\\caption{' + caption + '}', '\\label{' + label + '}', '\\centering', '\\setlength{\\tabcolsep}{3pt}', prefix,
             '\\begin{tabular}{@{}' + columns + '@{}}', '\\toprule', ' & '.join(header) + '\\\\', '\\midrule']
    lines += [' & '.join(str(value) for value in row) + '\\\\' for row in rows]
    lines += ['\\bottomrule', '\\end{tabular}', '\\widetableend' if wide else '\\end{' + environment + '}']
    path.write_text('\n'.join(lines) + '\n')


def export_figure(fig, name, width, height, directory):
    fig.set_size_inches(width, height)
    fig.savefig(directory / (name + '.png'), dpi=360)
    plt.close(fig)


def plotting_font():
    source = Path(subprocess.check_output(['kpsewhich', 'utmr8a.pfb'], text=True).strip())
    original = T1Font(str(source))
    glyphs = original.getGlyphSet()
    order = ['.notdef'] + [name for name in glyphs if name != '.notdef']
    charstrings, metrics = {}, {}
    for name in order:
        glyph = glyphs[name]
        glyph.draw(T2CharStringPen(0, None))
        width = round(glyph.width)
        pen = T2CharStringPen(width, None)
        glyph.draw(pen)
        charstrings[name] = pen.getCharString()
        metrics[name] = (width, 0)
    cmap = {ord(toUnicode(name)): name for name in order if len(toUnicode(name)) == 1}
    font = FontBuilder(1000, isTTF=False)
    font.setupGlyphOrder(order)
    font.setupCharacterMap(cmap)
    font.setupCFF('NimbusRomNo9L-Regu', {'FullName': 'Nimbus Roman No9 L Regular', 'FamilyName': 'Nimbus Roman No9 L', 'Weight': 'Regular', 'FontBBox': original['FontBBox']}, charstrings, {})
    font.setupHorizontalMetrics(metrics)
    font.setupHorizontalHeader(ascent=800, descent=-200)
    font.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=1000, usWinDescent=400)
    font.setupNameTable({'familyName': 'Nimbus Roman No9 L', 'styleName': 'Regular', 'uniqueFontIdentifier': 'NimbusRomNo9L-Regu-local-conversion', 'fullName': 'Nimbus Roman No9 L Regular', 'psName': 'NimbusRomNo9L-Regu'})
    font.setupPost()
    font.font['head'].created = font.font['head'].modified = 2082844800
    font.font.recalcTimestamp = False
    output = ROOT / 'report/fonts/NimbusRomanNo9L-Regular.otf'
    output.parent.mkdir(parents=True, exist_ok=True)
    font.save(output)
    font_manager.fontManager.addfont(str(output))
    return font_manager.FontProperties(fname=str(output)).get_name(), output, source


def main():
    completed = read(ROOT / 'output/verification/completed_model_runs_review.json')
    audit = read(ROOT / completed['complete_primary_audit'])
    cost_audit = read(ROOT / completed['cost_audit'])
    require(digest(ROOT / completed['complete_primary_audit']) == completed['complete_primary_audit_sha256'], 'Primary audit changed')
    require(digest(ROOT / completed['cost_audit']) == completed['cost_audit_sha256'], 'Cost audit changed')
    require(audit['primary_comparison_ready_for_report'] and cost_audit['models_verified'] == 25, 'Complete audits required')
    sources = {}
    font, font_path, type1_path = plotting_font()
    sources[str(font_path.relative_to(ROOT))] = digest(font_path)

    def source(path, expected=None):
        path = Path(path)
        value = digest(path)
        require(expected is None or value == expected, 'Audited asset input changed: ' + str(path))
        sources[str(path.relative_to(ROOT))] = value
        return read(path)

    protocol = source(ROOT / 'configs/primary_protocol.json')
    protocol_seal = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(audit['protocol_sha256'] == protocol_seal == (ROOT / 'configs/primary_protocol.sha256').read_text().strip(), 'Protocol differs')
    summary_path = PRIMARY / 'comparison_summary.json'
    summary = source(summary_path, audit['verified_files_sha256'][str(summary_path.relative_to(ROOT))])
    lock_path = PRIMARY / 'selection_lock.json'
    lock = source(lock_path, audit['verified_files_sha256'][str(lock_path.relative_to(ROOT))])
    spec = source(ROOT / 'configs/comparable_cost_protocol.json')
    cost_path = ROOT / spec['output_root'] / cost_audit['cost_protocol_sha256'][:16] / 'summary.json'
    cost = source(cost_path, cost_audit['summary_sha256'])
    historical_path = ROOT / 'materials/assignment2_baseline_records/selected_parameters.json'
    provenance = source(ROOT / 'output/verification/gate1_provenance_audit.json')
    historical_hash = '32a5c189e0326a0f170c90d84cc01a77bd61ae7c7cad47defd0edbae8e652e98'
    historical_provenance = [entry for entry in provenance['manifest_checks'] if entry['path'] == str(historical_path) and entry['role'] == 'copy']
    require(len(historical_provenance) == 1 and historical_provenance[0]['matches_manifest'] and historical_provenance[0]['sha256'] == historical_hash, 'Historical provenance seal differs')
    historical = source(historical_path, historical_hash)
    require([row['outer_fold'] for row in historical] == list(range(1, 6)), 'Historical fold coverage differs')
    for row in historical:
        for model, key in [('knn', 'knn'), ('classification_tree', 'classification_tree')]:
            matched = [record['identity']['setting'] for record in cost['records'] if record['identity']['fold'] == row['outer_fold'] and record['identity']['model'] == model]
            require(matched == [row[key]], 'Historical controls differ from audited predictive reproduction')
    labels, names, seeds = protocol['data']['class_order'], protocol['data']['class_names'], protocol['ensemble']['seeds']
    raw_path = ROOT / protocol['data']['file']
    split_path = ROOT / protocol['splits']['source']
    for path, expected in [(raw_path, protocol['data']['sha256']), (split_path, protocol['splits']['artifact_sha256'])]:
        require(digest(path) == expected, 'Partition source differs: ' + str(path))
        sources[str(path.relative_to(ROOT))] = expected
    targets = pd.read_csv(raw_path, usecols=[protocol['data']['target']])[protocol['data']['target']].to_numpy()
    require(len(targets) == protocol['data']['rows'], 'Partition target length differs')
    partition_rows = []
    roles = [('outer_train', 'Outer fitting'), ('outer_test', 'Outer test'), ('inner_train', 'Inner fitting'), ('inner_validation', 'Inner validation')]
    with np.load(split_path, allow_pickle=False) as saved:
        for index_seal in protocol['splits']['fold_index_hashes']:
            fold = index_seal['fold']
            for key, title in roles:
                indices = saved[f'fold_{fold}_{key}']
                require(hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest() == index_seal[key], 'Ordered partition indices differ')
                partition_rows.append({'fold': fold, 'role': title, 'rows': len(indices),
                                       'worms': int(np.count_nonzero(targets[indices] == names.index('Worms')))})
    partitions = pd.DataFrame(partition_rows)
    partition_ranges = [{'role': title, 'rows_min': int(partitions[partitions.role == title].rows.min()),
                         'rows_max': int(partitions[partitions.role == title].rows.max()),
                         'worms_min': int(partitions[partitions.role == title].worms.min()),
                         'worms_max': int(partitions[partitions.role == title].worms.max())} for _, title in roles]
    structural = []
    count_rows = []
    count_runs = []
    for path in sorted(PRIMARY.glob('selection/fold_*/leaf*/seed_*/record.json')):
        record = source(path, audit['verified_files_sha256'][str(path.relative_to(ROOT))])
        identity, measure, diversity = record['identity'], record['metrics'], record['diversity']
        structural.append({'fold': identity['fold'], 'seed': identity['seed'], 'candidate': identity['candidate_id'],
                           'leaf': identity['setting']['min_samples_leaf'], 'features': 'sqrt' if identity['setting']['max_features'] == 'sqrt' else 'all',
                           'weight': 'None' if identity['setting']['class_weight'] is None else 'Balanced',
                           **{k: measure[k] for k in ['macro_f1', 'balanced_accuracy', 'normal_false_alarm_rate']},
                           **{k: diversity[k] for k in ['mean_pairwise_disagreement', 'mean_member_error', 'mean_pairwise_simultaneous_error']},
                           **{f'{field}_{c}': measure[field][c] for field in ['precision', 'recall', 'f1'] for c in labels}})
    require(len(structural) == 180, 'Structural table requires all candidates and seeds')
    structure = pd.DataFrame(structural)
    for selected in lock['selections']:
        fold = selected['fold']
        for trees in [50, 100, 200]:
            runs = []
            for seed in seeds:
                sub = selected['candidate_id'] if trees == 100 else 'tree_count'
                path = PRIMARY / f'selection/fold_{fold}' / sub / f'seed_{seed}_n{trees}' / 'record.json'
                record = source(path, audit['verified_files_sha256'][str(path.relative_to(ROOT))])
                run = {'fold': fold, 'seed': seed, 'trees': trees, 'macro_f1': record['metrics']['macro_f1'],
                       'fit_seconds': None if record['fit_cost'] is None else record['fit_cost']['elapsed_seconds'],
                       'checkpoint_mib': record['checkpoint_bytes'] / 1024 ** 2}
                runs.append(run)
                count_runs.append(run)
            count_rows.append({'fold': fold, 'trees': trees, 'mean': float(np.mean([r['macro_f1'] for r in runs])),
                               'minimum': min(r['macro_f1'] for r in runs), 'maximum': max(r['macro_f1'] for r in runs),
                               'seed_sd': float(np.std([r['macro_f1'] for r in runs], ddof=1)),
                               'checkpoint_mib': float(np.mean([r['checkpoint_mib'] for r in runs])),
                               'fit_seconds': None if trees == 50 else float(np.mean([r['fit_seconds'] for r in runs]))})
    counts = pd.DataFrame(count_rows)
    outer = pd.DataFrame(summary['seed_fold_records'])
    aggregate = {'Ensemble': fold_summary(outer, ['macro_f1', 'balanced_accuracy', 'accuracy', 'normal_false_alarm_rate'])}
    for key, title in [('kNN', 'Neighbours'), ('Classification tree', 'Single tree')]:
        values = pd.DataFrame([{'fold': int(fold), **record} for fold, record in summary['baselines'][key]['folds'].items()])
        aggregate[title] = fold_summary(values, ['macro_f1', 'balanced_accuracy', 'accuracy', 'normal_false_alarm_rate'])
    class_rows = []
    for c in labels:
        row = {'class': names[c], 'label': c, 'support': summary['pooled_by_seed'][str(seeds[0])]['support'][c]}
        for model, data in [('ensemble', list(summary['pooled_by_seed'].values())), ('tree', [summary['baselines']['Classification tree']['pooled']]), ('neighbours', [summary['baselines']['kNN']['pooled']])]:
            for metric in ['precision', 'recall', 'f1']:
                row[f'{model}_{metric}'] = float(np.mean([d[metric][c] for d in data]))
        row['difference_tree'] = row['ensemble_f1'] - row['tree_f1']
        row['difference_neighbours'] = row['ensemble_f1'] - row['neighbours_f1']
        class_rows.append(row)
    classes = pd.DataFrame(class_rows)
    cost_rows = []
    for record in cost['records']:
        identity = record['identity']
        require(record['prediction_disagreements'] == 0, 'Cost comparison requires exact predictive reproduction')
        values = [record[k] for k in ['preprocessing_cost', 'fit_cost', 'prediction_cost', 'serialisation_cost']]
        cost_rows.append({'fold': identity['fold'], 'model': identity['model'], 'seed': identity['seed'],
                          **{name: record[name + '_cost']['elapsed_seconds'] for name in ['preprocessing', 'fit', 'prediction', 'serialisation']},
                          'checkpoint_mib': record['model_checkpoint_bytes'] / 1024 ** 2, 'encoder_kib': record['encoder_bytes'] / 1024,
                          'peak_gib': max(value['sampled_process_peak_rss_bytes'] for value in values) / 1024 ** 3})
    costs = pd.DataFrame(cost_rows)
    cost_summary = {model: fold_summary(frame, ['preprocessing', 'fit', 'prediction', 'serialisation', 'checkpoint_mib', 'encoder_kib', 'peak_gib']) for model, frame in costs.groupby('model')}
    factor = structure.groupby(['leaf', 'features', 'weight']).mean(numeric_only=True).drop(columns=['fold', 'seed']).reset_index()
    structural_folds = structure.groupby(['leaf', 'features', 'weight', 'fold'])[['macro_f1', 'balanced_accuracy']].mean()
    deviations = structural_folds.groupby(['leaf', 'features', 'weight']).std(ddof=1).add_suffix('_fold_sd').reset_index()
    factor = factor.merge(deviations, on=['leaf', 'features', 'weight'], validate='one_to_one')
    pairs = []
    for key, frame in structure.groupby(['fold', 'seed', 'leaf', 'weight']):
        all_features, sampled = frame.set_index('features').loc['all'], frame.set_index('features').loc['sqrt']
        pairs.append(dict(zip(['fold', 'seed', 'leaf', 'weight'], key)) | {metric: float(sampled[metric] - all_features[metric]) for metric in ['macro_f1', 'mean_pairwise_disagreement', 'mean_member_error', 'mean_pairwise_simultaneous_error']})
    paired_features = pd.DataFrame(pairs)
    pooled = [{'pipeline': 'Neighbours', **summary['baselines']['kNN']['pooled']}, {'pipeline': 'Single tree', **summary['baselines']['Classification tree']['pooled']}]
    pooled += [{'pipeline': 'Ensemble ' + str(seed), **summary['pooled_by_seed'][str(seed)]} for seed in seeds]
    confusion = {'neighbours': np.asarray(summary['baselines']['kNN']['pooled']['row_normalised_confusion']),
                 'tree': np.asarray(summary['baselines']['Classification tree']['pooled']['row_normalised_confusion']),
                 'ensemble': np.mean([v['row_normalised_confusion'] for v in summary['pooled_by_seed'].values()], axis=0)}
    annotations = {model: confusion_annotations(matrix) for model, matrix in confusion.items()}
    analysis = {'status': 'audited_inputs_aggregated_no_new_selection', 'aggregate': aggregate, 'classes': class_rows,
                'partitions': partition_rows, 'partition_ranges': partition_ranges, 'confusion_annotations': annotations,
                'confusion_annotation_rule': {'cells': 'Diagonal and largest positive off-diagonal per true-class row; ties use first class in fixed order',
                                              'units': 'Whole percentages; positive values below 0.5 percent displayed as <1',
                                              'unlabelled': 'Colour retained; absence of text does not mean zero', 'font_points': 10},
                'structural': factor.to_dict(orient='records'), 'tree_count': count_rows, 'pooled': pooled, 'costs': cost_summary,
                'paired': summary['paired_comparisons'], 'selected': lock['selections'], 'historical_selected': historical,
                'feature_paired_differences': paired_features.to_dict(orient='records'),
                'count_mean_across_folds': counts.groupby('trees')['mean'].mean().to_dict(),
                'count_additional_fit_seconds': counts[counts.trees == 200].fit_seconds.describe().to_dict(),
                'clock_limitations': cost_audit['timing_observations'], 'class_support_total': int(classes.support.sum()),
                'role': 'Reporting of frozen primary and predeclared supplementary evidence; no new fitting, control selection or formal test'}
    identity = hashlib.sha256(json.dumps({'source': sources, 'code': digest(__file__)}, sort_keys=True).encode()).hexdigest()[:16]
    archive = ROOT / 'output/report_assets' / identity
    archive.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, frame in [('structural_runs', structure), ('structural_means', factor), ('tree_count_runs', pd.DataFrame(count_runs)), ('tree_count_means', counts), ('outer_seed_fold', outer), ('class_assessment', classes), ('cost_runs', costs), ('feature_paired_contrasts', paired_features)]:
        frame.to_csv(archive / (name + '.csv'), index=False)
    (archive / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    partitions.to_csv(archive / 'partition_counts.csv', index=False)
    table(GENERATED / 'partitions.tex', 'Partition sizes across five folds', 'tab:partitions', ['Role', 'Rows', 'Worms'],
          [[row['role'], f"{row['rows_min']}--{row['rows_max']}", f"{row['worms_min']}--{row['worms_max']}"] for row in partition_ranges])
    table(GENERATED / 'historical_neighbours.tex', 'Selected historical neighbour controls', 'tab:historical-neighbours', ['Fold', '$k$', 'Distance', 'Vote', '$q$'],
          [[row['outer_fold'], row['knn']['k'], 'Manhattan', 'Uniform' if row['knn']['weights'] == 'uniform' else 'Distance', '--' if row['knn']['weights'] == 'uniform' else f"{row['knn']['vote_exponent']:g}"] for row in historical])
    table(GENERATED / 'historical_tree.tex', 'Selected historical tree controls', 'tab:historical-tree', ['Fold', 'Criterion', 'Depth', 'Leaf', 'Weight', '$\\lambda$'],
          [[row['outer_fold'], row['classification_tree']['criterion'].capitalize(), row['classification_tree']['max_depth'], row['classification_tree']['min_samples_leaf'], 'None' if row['classification_tree']['class_weight'] is None else 'Balanced', '$10^{' + str(int(np.log10(row['classification_tree']['ccp_alpha']))) + '}$'] for row in historical])
    table(GENERATED / 'selected.tex', 'Selected ensemble controls', 'tab:selected', ['Fold', 'Leaf size', 'Trees', 'Inner score'],
          [[s['fold'], s['setting']['min_samples_leaf'], s['trees'], f"{s['count_selection']['macro_f1']:.4f}"] for s in lock['selections']])
    table(GENERATED / 'structural.tex', 'Structural validation assessment', 'tab:structural', ['Leaf', 'Coordinates', 'Weight', '$f_{\\mathrm{m}}$', 'Deviation', '$\\beta$'],
          [[int(a.leaf), 'All' if a.features == 'all' else '13', a.weight, f'{a.macro_f1:.4f}', f'{a.macro_f1_fold_sd:.4f}', f'{a.balanced_accuracy:.4f}'] for a in factor.itertuples()])
    table(GENERATED / 'counts.tex', 'Conditional validation scores', 'tab:counts', ['Fold', '50 trees', '100 trees', '200 trees'],
          [[fold] + [f"{counts[(counts.fold == fold) & (counts.trees == n)]['mean'].iloc[0]:.4f} $\\pm$ {counts[(counts.fold == fold) & (counts.trees == n)]['seed_sd'].iloc[0]:.4f}" for n in [50, 100, 200]] for fold in range(1, 6)])
    table(GENERATED / 'aggregate.tex', 'Matched outer assessment', 'tab:aggregate', ['Pipeline', '$f_{\\mathrm{m}}$', 'Deviation', '$\\beta$', 'False alarms'],
          [[title, f"{aggregate[title]['macro_f1']['mean']:.4f}", f"{aggregate[title]['macro_f1']['fold_sd']:.4f}", f"{aggregate[title]['balanced_accuracy']['mean']:.4f}", f"{aggregate[title]['normal_false_alarm_rate']['mean']:.4f}"] for title in ['Neighbours', 'Single tree', 'Ensemble']])
    table(GENERATED / 'paired.tex', 'Paired outer score differences', 'tab:paired', ['Fold', 'Minus neighbours', 'Minus tree', 'Seed deviation'],
          [[fold, f"+{summary['paired_comparisons']['kNN'][fold - 1]['difference']:.4f}", f"+{summary['paired_comparisons']['Classification tree'][fold - 1]['difference']:.4f}", f"{summary['folds'][fold - 1]['macro_f1_seed_sd']:.4f}"] for fold in range(1, 6)])
    table(GENERATED / 'pooled.tex', 'Separately pooled assessment', 'tab:pooled', ['Pipeline', '$f_{\\mathrm{m}}$', '$\\beta$', 'Accuracy', 'False alarms'],
          [[p['pipeline']] + [f'{p[name]:.4f}' for name in ['macro_f1', 'balanced_accuracy', 'accuracy', 'normal_false_alarm_rate']] for p in pooled])
    class_header = ['Class', 'Support'] + ['$\\pi_c$', '$r_c$', '$f_{1,c}$'] * 3
    class_table_rows = [[r['class'], str(r['support'])] + [f"{r[model + '_' + metric]:.4f}" for model in ['neighbours', 'tree', 'ensemble'] for metric in ['precision', 'recall', 'f1']] for r in class_rows]
    group = '\\multicolumn{2}{c}{} & \\multicolumn{3}{c}{Nearest neighbours} & \\multicolumn{3}{c}{Single tree} & \\multicolumn{3}{c}{Tree ensemble}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n'
    table(GENERATED / 'classes.tex', 'Per-class pooled assessment', 'tab:class-results', class_header, class_table_rows, wide=True)
    content = (GENERATED / 'classes.tex').read_text().replace('\\toprule\n', '\\toprule\n' + group, 1)
    (GENERATED / 'classes.tex').write_text(content)
    for filename, caption, label, headers, fields in [
        ('costs', 'Comparable operation times', 'tab:costs', ['Preprocess', 'Fit', 'Predict', 'Persist'], ['preprocessing', 'fit', 'prediction', 'serialisation']),
        ('storage', 'Comparable storage and process memory', 'tab:storage', ['\\shortstack{Model\\\\mebibytes}', '\\shortstack{Encoder\\\\kibibytes}', '\\shortstack{Peak\\\\gibibytes}'], ['checkpoint_mib', 'encoder_kib', 'peak_gib'])]:
        table(GENERATED / (filename + '.tex'), caption, label, ['Pipeline'] + headers,
              [[title] + [f"{cost_summary[model][metric]['mean']:.3f} $\\pm$ {cost_summary[model][metric]['fold_sd']:.3f}" for metric in fields] for model, title in [('knn', 'Neighbours'), ('classification_tree', 'Single tree'), ('forest', 'Ensemble')]], wide=True)
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'mathtext.fontset': 'stix', 'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange = '#135B8A', '#B44E0B'
    minimum, maximum = counts.minimum.min() - 0.002, counts.maximum.max() + 0.002
    for fold in range(1, 6):
        data = counts[counts.fold == fold].sort_values('trees')
        fig, ax = plt.subplots(layout='constrained')
        ax.errorbar(data.trees, data['mean'], yerr=np.vstack([data['mean'] - data.minimum, data.maximum - data['mean']]), color=blue, marker='o', markersize=4, capsize=3, linewidth=1)
        ax.set(xlabel='Trees', ylabel='Mean class score', xticks=[50, 100, 200], ylim=(minimum, maximum))
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid(alpha=.18)
        export_figure(fig, f'count_{fold}', 2.29, 1.9, GENERATED)
    for model, short in [('kNN', 'neighbours'), ('Classification tree', 'tree')]:
        fig, ax = plt.subplots(layout='constrained')
        delta = summary['paired_comparisons'][model]
        means = np.array([a['difference'] for a in delta])
        low, high = [], []
        for a in delta:
            values = outer[outer.fold == a['fold']].macro_f1.to_numpy() - a['comparator_macro_f1']
            low.append(values.min()); high.append(values.max())
        ax.axhline(0, color='.45', linewidth=.8)
        ax.errorbar(range(1, 6), means, yerr=np.vstack([means - low, high - means]), marker='o', markersize=4, color=blue, capsize=4, linewidth=1)
        ax.set(xlabel='Outer fold', ylabel='Class-score difference', xticks=range(1, 6))
        ax.set_ylim(min(-.005, min(low) - .005), max(high) + .005)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=.18)
        export_figure(fig, 'paired_' + short, 3.43, 2.45, GENERATED)
    for model, matrix in confusion.items():
        fig, ax = plt.subplots(layout='constrained')
        picture = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='equal')
        for label in annotations[model]:
            ax.text(label['predicted'], label['true'], label['text'], ha='center', va='center', fontsize=10,
                    color='white' if label['proportion'] >= .55 else 'black')
        ax.set_xticks(range(10), names, rotation=65, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(10), names)
        ax.set(xlabel='Predicted class', ylabel='True class')
        fig.colorbar(picture, ax=ax, fraction=.035, pad=.025, ticks=[0, .5, 1])
        export_figure(fig, 'confusion_' + model, 3.43, 3.55, GENERATED)
    for weight in ['None', 'Balanced']:
        fig, ax = plt.subplots(layout='constrained')
        for feature, color, marker, title in [('all', blue, 'o', 'All coordinates'), ('sqrt', orange, 's', '13 candidates')]:
            data = factor[(factor.weight == weight) & (factor.features == feature)].sort_values('leaf')
            ax.plot(data.mean_pairwise_disagreement, data.macro_f1, marker=marker, markersize=4, color=color, linewidth=.8, label=title)
            for a in data.itertuples():
                offset = (-13, -13) if a.leaf == 1 else (5, 7) if a.leaf == 5 else (5, 5)
                ax.annotate(str(int(a.leaf)), (a.mean_pairwise_disagreement, a.macro_f1), xytext=offset, textcoords='offset points', fontsize=10)
        ax.set(xlabel='Member disagreement', ylabel='Mean class score')
        ax.set_xlim(factor.mean_pairwise_disagreement.min() - .015, factor.mean_pairwise_disagreement.max() + .025)
        ax.set_ylim(factor.macro_f1.min() - .025, factor.macro_f1.max() + .045)
        ax.legend(loc='lower right', frameon=False)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(alpha=.18)
        export_figure(fig, 'diversity_' + weight.lower(), 3.43, 2.65, GENERATED)
    for path in GENERATED.iterdir():
        if path.suffix in ['.png', '.tex']:
            shutil.copy2(path, archive / path.name)
    shutil.copy2(__file__, archive / 'make_report_assets.py')
    manifest = {'asset_identity': identity, 'script_sha256': digest(__file__), 'input_hashes': sources, 'font': font,
                'font_source': {'type1_path': str(type1_path), 'type1_sha256': digest(type1_path), 'converted_sha256': digest(font_path), 'purpose': 'Original report-body Type 1 outlines converted to an OpenType container for matching plot text; no plot font substitution'},
                'software': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
                'files': {path.name: digest(path) for path in sorted(archive.iterdir()) if path.is_file()},
                'new_fits': 0, 'seed_selection': False, 'no_formal_test_or_independent_replicate_claim': True}
    (archive / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (GENERATED / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'generated_from_audited_evidence', 'directory': str(archive), 'tables': 12, 'figure_panels': 12, 'font': font}))


if __name__ == '__main__':
    main()
