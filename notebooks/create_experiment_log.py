
from pathlib import Path

import nbformat as nbf


cells = []



def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


code("print('ML441 Assignment 3 staged experiment log')")

code("""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "configs/primary_protocol.json").exists():
    ROOT = ROOT.parent
if not (ROOT / "configs/primary_protocol.json").exists():
    raise RuntimeError("Start Jupyter in the assignment3 project or notebooks directory")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
EXPORT = ROOT / "notebooks/exports"
EXPORT.mkdir(parents=True, exist_ok=True)
AUTO_COMMIT = os.environ.get("ML441_NOTEBOOK_COMMIT", "1") == "1"
PUSH_TO_REMOTE = os.environ.get("ML441_NOTEBOOK_PUSH", "0") == "1"
protocol = json.loads((ROOT / "configs/primary_protocol.json").read_text())

def run_module(name, *arguments):
    command = [sys.executable, "-m", name, *arguments]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print("$", " ".join(command))
    print(result.stdout[-1500:])
    if result.returncode:
        print(result.stderr[-3000:])
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    return result

def publish_stage(label, paths):
    if not AUTO_COMMIT:
        print("Review stage before committing:", label, paths)
        return
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode != 0:
        raise RuntimeError("Commit or unstage existing staged changes before publishing this stage")
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    changes = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if changes.returncode:
        subprocess.run(["git", "commit", "-m", label], cwd=ROOT, check=True)
    else:
        print("No new changes for", label)
    if PUSH_TO_REMOTE:
        remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                                text=True, capture_output=True)
        if remote.returncode:
            raise RuntimeError("Configure a Git remote named origin before enabling pushes")
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)

print("Repository:", ROOT)
print("Local commits enabled:", AUTO_COMMIT, "| remote pushes enabled:", PUSH_TO_REMOTE)
""")

code("print('Stage 1: Source snapshots and unit tests')")

code("""
from src.download_data import SOURCES, sha256
from src.data import load_series

manifest = []
for key, filename in zip(protocol["datasets"], protocol["raw_sha256"]):
    path = ROOT / "data/raw" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen source: {path}")
    actual_hash = sha256(path)
    expected_hash = protocol["raw_sha256"][filename]
    if actual_hash != expected_hash:
        raise ValueError(f"Source snapshot changed: {filename}")
    series = load_series(key)
    manifest.append({"dataset": key, "local_file": str(path.relative_to(ROOT)),
                     "source_url": SOURCES[filename], "sha256": actual_hash,
                     "target": series.source_note, "unit": series.unit,
                     "frequency": "day" if key == "power" else "month",
                     "modelled_length": len(series.values),
                     "modelled_missing": int(series.levels.isna().sum())})
manifest_table = pd.DataFrame(manifest)
manifest_table.to_csv(EXPORT / "dataset_manifest.csv", index=False)
display(manifest_table)
test_result = run_module("pytest", "-q", "tests")
passed = re.search(r"(\\d+) passed", test_result.stdout)
if passed is None:
    raise AssertionError("Pytest did not report a passing test count")
(EXPORT / "unit_tests.json").write_text(json.dumps({
    "command": "python -m pytest -q tests", "exit_code": test_result.returncode,
    "passed_tests": int(passed.group(1))}, indent=2))
""")

code("""
publish_stage("Archive frozen sources and unit-test evidence", [
    "data/raw", "notebooks/exports/dataset_manifest.csv",
    "notebooks/exports/unit_tests.json"])
""")

code("print('Stage 2: Development pilot')")

code("""
if not (ROOT / "output/pilot/summary.json").exists():
    run_module("src.pilot")
pilot = json.loads((ROOT / "output/pilot/summary.json").read_text())
if not pilot["deterministic_repeat"]:
    raise AssertionError("Pilot repeat was not deterministic")
(EXPORT / "pilot_summary.json").write_text(json.dumps(pilot, indent=2))
display(pd.DataFrame(pilot["runs"]))
""")

code("""
publish_stage("Archive development-only pilot", [
    "output/pilot", "notebooks/exports/pilot_summary.json"])
""")

code("print('Stage 3: Development data audit')")

code("""
if not (ROOT / "output/diagnostics/data_audit.json").exists():
    run_module("src.audit_data")
audit = json.loads((ROOT / "output/diagnostics/data_audit.json").read_text())
audit_table = pd.DataFrame([{
    "dataset": row["dataset"], "modelled_n": row["modelled_n"],
    "test_start": row["test_start"],
    "development_missing": row["development_missing"],
    "level_adf_p": row["development_level"]["adf_unit_root_null_p"],
    "level_kpss_p": row["development_level"]["kpss_level_stationarity_null_p"]}
    for row in audit])
audit_table.to_csv(EXPORT / "development_audit.csv", index=False)
display(audit_table)
""")

code("""
publish_stage("Archive development-only data audit", [
    "output/diagnostics/data_audit.json", "notebooks/exports/development_audit.csv"])
""")

code("print('Stage 4: Primary development selection')")

code("""
run_module("src.run_experiment", "--phase", "selection")
freeze = json.loads((ROOT / "output/primary/all_selected_before_test.json").read_text())
validation_rows = []
selected_rows = []
for dataset in protocol["datasets"]:
    folder = ROOT / "output/primary/selection" / dataset
    for path in sorted(folder.glob("*_f*_s*.json")):
        row = json.loads(path.read_text())
        validation_rows.append({
            "dataset": dataset, "architecture": row["architecture"],
            "hidden": row["hidden"], "learning_rate": row["learning_rate"],
            "fold": row["fold"], "seed": row["seed"],
            "mae": row["dataset_metrics"]["mae"],
            "rmse": row["dataset_metrics"]["rmse"],
            "mase": row["dataset_metrics"]["mase"],
            "best_epoch": row["best_epoch"],
            "parameters": row["parameter_count"],
            "scaler_fitted_end": row["scaler"]["fitted_end"]})
    for architecture, choice in freeze[dataset]["selected"].items():
        selected_rows.append({"dataset": dataset, "architecture": architecture,
                              **choice})
if len(validation_rows) != 540 or len(selected_rows) != 15:
    raise AssertionError("Incomplete development selection")
pd.DataFrame(validation_rows).to_csv(EXPORT / "all_validation_runs.csv", index=False)
selected_table = pd.DataFrame(selected_rows)
selected_table.to_csv(EXPORT / "selected_configurations.csv", index=False)
display(selected_table)
print("Saved", len(validation_rows), "validation records and", len(selected_rows), "selections")
""")

code("""
publish_stage("Archive frozen validation selection", [
    "output/primary/protocol_snapshot.json", "output/primary/selection",
    "output/primary/all_selected_before_test.json",
    "notebooks/exports/all_validation_runs.csv",
    "notebooks/exports/selected_configurations.csv"])
""")

code("print('Stage 5: Final chronological evaluation')")

code("""
if not (ROOT / "output/primary/all_selected_before_test.json").exists():
    raise RuntimeError("Selection must be frozen before final evaluation")
run_module("src.run_experiment", "--phase", "evaluation")
final_rows = []
for dataset in protocol["datasets"]:
    for path in sorted((ROOT / "output/primary/evaluation" / dataset).glob("*_s*.json")):
        row = json.loads(path.read_text())
        final_rows.append({"dataset": dataset, "architecture": row["architecture"],
                           "seed": row["seed"], "hidden": row["hidden"],
                           "learning_rate": row["learning_rate"],
                           "parameters": row["parameter_count"],
                           "epochs": row["epochs"],
                           "targets": len(row["indices"]),
                           **row["dataset_metrics"]})
if len(final_rows) != 45:
    raise AssertionError("Incomplete final evaluation")
final_table = pd.DataFrame(final_rows)
final_table.to_csv(EXPORT / "all_final_seed_runs.csv", index=False)
summaries = json.loads((ROOT / "output/primary/summary.json").read_text())
baseline_rows = [{"dataset": row["dataset"], "rule": rule, **metrics}
                 for row in summaries for rule, metrics in row["baselines"].items()]
pd.DataFrame(baseline_rows).to_csv(EXPORT / "all_baselines.csv", index=False)
display(final_table.groupby(["dataset", "architecture"])["mae"].agg(["mean", "std"]))
print("Saved", len(final_rows), "seed-level final results")
""")

code("""
publish_stage("Archive final tests and diagnostic baselines", [
    "output/primary/evaluation", "output/primary/summary.json",
    "notebooks/exports/all_final_seed_runs.csv",
    "notebooks/exports/all_baselines.csv"])
""")

code("print('Stage 6: Independent verification and diagnostics')")

code("""
run_module("src.verify_results")
run_module("src.diagnose_training")
verification = json.loads((ROOT / "output/diagnostics/verification.json").read_text())
curves = json.loads((ROOT / "output/diagnostics/training_curves.json").read_text())
display(pd.DataFrame([{
    "verification": verification["status"],
    "selection_records": verification["selection_records"],
    "final_records": verification["final_network_records"],
    "selected_development_fits": curves["selected_development_fits"],
    "training_loss_lower_after_checkpoint":
        curves["training_loss_lower_after_selected_checkpoint"]}]))
""")

code("""
publish_stage("Archive verification diagnostics and report", [
    "output/diagnostics/verification.json", "output/diagnostics/training_curves.json",
    "report/main.tex", "report/generated", "report/27187314RW441assignment3.pdf"])
""")

code("print('Remote upload: Push reviewed commits only')")


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
destination = Path(__file__).with_name("experiment_log.ipynb")
nbf.write(notebook, destination)
print(destination)
