
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_series
from .splits import boundaries


DEST = Path("report/generated")
PRIMARY = Path("output/primary")


def fmt_p(value: float) -> str:
    if value <= 0.0100001:
        return "$\\leq 0.01$"
    if value >= 0.099999:
        return "$\\geq 0.10$"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write(name: str, content: str) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(content)


def static_tables() -> None:
    audit = json.loads(Path("output/diagnostics/data_audit.json").read_text())
    labels = {"co2": "Carbon dioxide", "sunspots": "Sunspots", "power": "Household power",
              "retail": "Retail volume", "rain": "Precipitation"}
    frequencies = {"co2": "month", "sunspots": "month", "power": "day",
                   "retail": "month", "rain": "month"}
    rows = []
    split_rows = []
    for row in audit:
        key = row["dataset"]
        missing = row["development_missing"] + row["test_missing_count_for_integrity_only"]
        rows.append(f"{labels[key]} & {row['modelled_n']:,} & {frequencies[key]} & {missing} & "
                    f"{row['test_start']} & {fmt_p(row['development_level']['adf_unit_root_null_p'])} & "
                    f"{fmt_p(row['development_level']['kpss_level_stationarity_null_p'])} \\\\")
        data = load_series(key)
        _, _, folds = boundaries(len(data.values))
        starts = [str(data.values.index[fold.stop_end].date()) for fold in folds]
        split_rows.append(f"{labels[key]} & {starts[0]} & {starts[1]} & {starts[2]} & "
                          f"{row['test_start']} & {row['test_n']} \\\\")
    write("data_table.tex", """\\begin{table*}[t]
\\caption{Selected series, source missingness, and development-period level tests}
\\label{tab:data}
\\centering\\small
\\begin{tabular}{lrrrlll}
\\toprule
Series & Length & Frequency & Missing & Test begins & ADF $p$ & KPSS $p$ \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("split_table.tex", """\\begin{table*}[t]
\\caption{Start dates of the three validation blocks and final test}
\\label{tab:splits}
\\centering\\small
\\begin{tabular}{lllllr}
\\toprule
Series & First block & Second block & Third block & Test & Test positions \\\\
\\midrule
""" + "\n".join(split_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("architecture_table.tex", r"""\begin{table}[t]
\caption{Feedback source and trainable parameters}
\label{tab:architectures}
\centering\small
\begin{tabular}{lrrr}
\toprule
Network & Feedback & Small & Large \\
\midrule
Elman & hidden & 89 & 379 \\
Jordan & output & 97 & 385 \\
Combined & both & 97 & 397 \\
\bottomrule
\end{tabular}
\end{table}
""")
    write("experiment_table.tex", r"""\begin{table*}[t]
\caption{Primary experiment roles and supplementary diagnostic roles}
\label{tab:experiments}
\centering\footnotesize
\begin{tabular}{lllll}
\toprule
Experiment & Partition & Varied factor & Selection role & Main limitation \\
\midrule
Pipeline pilot & CO$_2$ development fold 1 & Network and seed & None & Pipeline check only \\
Control search & Three validation blocks & Capacity and rate & Select settings & Finite grid \\
Final comparison & Final chronological test & Network and seed & Conclusion only & One test period \\
Baseline comparison & Same final targets & Forecast rule & Interpretation only & Reference choice \\
Training curves & Development histories & Epoch & Explanation only & Descriptive only \\
Paired errors, post-hoc & Frozen final predictions & Target and seed & None & Dependent origins \\
Budget sensitivity, post-hoc & Saved development folds & Budget and rate & None & Reuses selection folds \\
Fit sensitivity, post-hoc & Saved development curves & Epoch & None & Different metrics \\
Trend sensitivity, post-hoc & Development observations & Test specification & None & Test assumptions \\
\bottomrule
\end{tabular}
\end{table*}
""")


def result_tables(summaries: list[dict]) -> None:
    labels = {"co2": "Carbon dioxide", "sunspots": "Sunspots", "power": "Power",
              "retail": "Retail", "rain": "Rainfall"}
    arch_labels = {"elman": "Elman", "jordan": "Jordan", "multi": "Combined"}
    rows = []
    for summary in summaries:
        key = summary["dataset"]
        selection = json.loads((PRIMARY / "selection" / key / "selected.json").read_text())
        for architecture in ["elman", "jordan", "multi"]:
            chosen = selection["selected"][architecture]
            rows.append(f"{labels[key]} & {arch_labels[architecture]} & {chosen['hidden']} & "
                        f"{chosen['parameter_count']} & {chosen['learning_rate']:.3f} & "
                        f"{chosen['mean_validation_mae']:.3f} & {chosen['epochs']} & "
                        f"{chosen['cap_limited_fits']} \\\\")
    write("selection_table.tex", """\\begin{table*}[t]
\\caption{Selected controls from expanding development validation}
\\label{tab:selection}
\\centering\\small
\\begin{tabular}{llrrrrrr}
\\toprule
Series & Network & Width & Parameters & Rate & Validation MAE & Refit epochs & Cap fits \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    rows = []
    baseline_rows = []
    for summary in summaries:
        key = summary["dataset"]
        for architecture in ["elman", "jordan", "multi"]:
            result = summary["networks"][architecture]
            rows.append(f"{labels[key]} & {arch_labels[architecture]} & "
                        f"${result['mean_mae']:.3f}\\pm{result['sd_mae']:.3f}$ & "
                        f"{result['mean_rmse']:.3f} & {result['mean_mase']:.3f} \\\\")
        baseline = summary["baselines"]
        seasonal = baseline.get("seasonal_naive")
        baseline_rows.append(f"{labels[key]} & {summary['test_target_count']} & "
                             f"{baseline['persistence']['mae']:.3f} & "
                             f"{seasonal['mae']:.3f} \\\\" if seasonal else
                             f"{labels[key]} & {summary['test_target_count']} & "
                             f"{baseline['persistence']['mae']:.3f} & -- \\\\")
    write("performance_table.tex", """\\begin{table*}[t]
\\caption{Final-period network errors averaged across three initialisation seeds}
\\label{tab:performance}
\\centering\\small
\\begin{tabular}{llrrr}
\\toprule
Series & Network & MAE $\\pm$ seed standard deviation & RMSE & MASE \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("baseline_table.tex", """\\begin{table}[t]
\\caption{Final-period diagnostic mean absolute error in original units}
\\label{tab:baselines}
\\centering\\footnotesize
\\begin{tabular}{lrrr}
\\toprule
Series & Targets & Persistence & Seasonal \\\\
\\midrule
""" + "\n".join(baseline_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def supplementary_tables() -> None:
    paired = json.loads(Path("output/diagnostics/paired_errors.json").read_text())
    capacity = json.loads(Path("output/diagnostics/capacity_sensitivity.json").read_text())
    fit = json.loads(Path("output/diagnostics/fit_sensitivity.json").read_text())
    labels = {"co2": "Carbon dioxide", "sunspots": "Sunspots", "power": "Power",
              "retail": "Retail", "rain": "Rainfall"}
    units = {"co2": "ppm", "sunspots": "count", "power": "kW",
             "retail": "index", "rain": "mm"}
    names = {"elman": "Elman", "jordan": "Jordan", "multi": "Combined"}
    rows = []
    for row in paired["results"]:
        key = row["dataset"]
        rows.append(f"{labels[key]} & {names[row['winner']]} & {names[row['runner']]} & "
                    f"{row['mean_error_advantage']:.4f} {units[key]} & "
                    f"{row['relative_advantage_percent']:.2f} & "
                    f"{row['paired_seed_wins']}/3 \\\\")
    write("paired_table.tex", """\\begin{table*}[t]
\\caption{Paired final-period error advantages of observed leaders}
\\label{tab:paired}
\\centering\\small
\\begin{tabular}{lllrrr}
\\toprule
Series & Leader & Next network & Mean advantage & Relative advantage (\\%) & Seed wins \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    rows = []
    for row in capacity["results"]:
        small, large = row["budgets"]
        rows.append(f"{labels[row['dataset']]} & {names[small['leader']]} & "
                    f"{small['leader_mean_validation_mae']:.3f} & "
                    f"{names[large['leader']]} & "
                    f"{large['leader_mean_validation_mae']:.3f} \\\\")
    write("capacity_table.tex", """\\begin{table*}[t]
\\caption{Development leaders within each trainable-parameter budget}
\\label{tab:capacity}
\\centering\\small
\\begin{tabular}{llrlr}
\\toprule
Series & Small leader & Validation error & Large leader & Validation error \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    rows = []
    for row in fit["results"]:
        rows.append(f"{labels[row['dataset']]} & {row['median_best_epoch']:.0f} & "
                    f"{row['median_train_reduction_to_checkpoint_percent']:.1f} & "
                    f"{row['median_tail_reduction_to_checkpoint_percent']:.1f} & "
                    f"{row['median_train_reduction_after_checkpoint_percent']:.1f} & "
                    f"{row['median_tail_increase_after_checkpoint_percent']:.1f} \\\\")
    write("fit_table.tex", """\\begin{table*}[t]
\\caption{Median development-fold training and stopping-tail changes}
\\label{tab:fit}
\\centering\\footnotesize
\\begin{tabular}{lrrrrr}
\\toprule
Series & Selected epoch & Training drop to best (\\%) & Tail drop to best (\\%) & Training drop after best (\\%) & Tail rise after best (\\%) \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def figures(summaries: list[dict]) -> None:
    keys = ["co2", "sunspots", "power", "retail", "rain"]
    labels = {"co2": "Carbon dioxide (ppm)", "sunspots": "Sunspot number",
              "power": "Power (kW)", "retail": "Retail volume index",
              "rain": "Rainfall (mm)"}
    fig, axes = plt.subplots(5, 1, figsize=(7.1, 5.7), constrained_layout=True)
    for ax, key in zip(axes, keys):
        data = load_series(key)
        development_end, _, _ = boundaries(len(data.values))
        ax.plot(data.levels.index, data.levels.values, color="#245082", linewidth=0.7)
        ax.axvline(data.levels.index[development_end], color="#c44e30", linewidth=1,
                   linestyle="--")
        ax.set_ylabel(labels[key], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    fig.savefig(DEST / "series.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 2.0), constrained_layout=True)
    x = np.arange(len(keys))
    width = 0.25
    colors = {"elman": "#245082", "jordan": "#cd7c2d", "multi": "#4e8a65"}
    for offset, architecture in enumerate(["elman", "jordan", "multi"]):
        ratios = [next(s for s in summaries if s["dataset"] == key)["networks"][architecture]["mean_mae"] /
                  next(s for s in summaries if s["dataset"] == key)["baselines"]["persistence"]["mae"]
                  for key in keys]
        ax.bar(x + (offset-1)*width, ratios, width, label={"elman":"Elman","jordan":"Jordan","multi":"Combined"}[architecture],
               color=colors[architecture])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, ["CO$_2$", "Sunspots", "Power", "Retail", "Rainfall"])
    ax.set_ylabel("MAE / persistence MAE", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(ncol=3, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(DEST / "skill.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.1), constrained_layout=True)
    for ax, key in zip(axes, ["co2", "retail"]):
        data = load_series(key)
        architecture = min(s for s in summaries if s["dataset"] == key)["networks"]
        architecture = min(architecture, key=lambda a: architecture[a]["mean_mae"])
        records = [json.loads((PRIMARY / "evaluation" / key / f"{architecture}_s{seed}.json").read_text())
                   for seed in [11, 23, 37]]
        indices = np.array(records[0]["indices"])
        forecast = np.mean([record["forecast"] for record in records], axis=0)
        tail = min(70, len(indices))
        dates = data.levels.index[indices[-tail:]]
        ax.plot(dates, np.array(records[0]["actual"])[-tail:], color="black", label="Observed", linewidth=1)
        ax.plot(dates, forecast[-tail:], color="#245082", label="Mean forecast", linewidth=1)
        ax.set_ylabel(labels[key], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=7)
    fig.savefig(DEST / "examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    static_tables()
    if not (PRIMARY / "summary.json").exists():
        print("Primary run incomplete; static tables only")
        return
    if not (PRIMARY / "all_selected_before_test.json").exists():
        raise RuntimeError("Missing pre-test selection freeze")
    summaries = json.loads((PRIMARY / "summary.json").read_text())
    result_tables(summaries)
    supplementary_tables()
    figures(summaries)
    print("Report tables and figures generated")


if __name__ == "__main__":
    main()
