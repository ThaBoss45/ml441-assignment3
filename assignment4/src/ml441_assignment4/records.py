import hashlib
import importlib.metadata
import json
import os
import platform
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import psutil


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def software_versions():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "psutil", "python-dateutil", "six", "narwhals"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def code_identity():
    directory = Path(__file__).parent
    return {path.name: file_hash(path) for path in sorted(directory.glob("*.py"))}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(directory, identity, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        load_checkpoint(directory, identity)
        return directory / "model.joblib"
    target = directory / "model.joblib"
    temporary = directory / "model.joblib.tmp"
    joblib.dump({"identity": identity, "payload": payload}, temporary, compress=0)
    os.replace(temporary, target)
    write_json(directory / "complete.json", {
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "model_sha256": file_hash(target),
        "model_bytes": target.stat().st_size,
    })
    return target


def load_checkpoint(directory, identity):
    directory = Path(directory)
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest["identity"] != identity or manifest["identity_sha256"] != canonical_hash(identity):
        raise ValueError("Incompatible checkpoint identity")
    path = directory / "model.joblib"
    if file_hash(path) != manifest["model_sha256"]:
        raise ValueError("Checkpoint content hash differs")
    saved = joblib.load(path)
    if saved["identity"] != identity:
        raise ValueError("Checkpoint embedded identity differs")
    return saved["payload"]


class MemoryMonitor:
    def __init__(self, limit_gib=6, interval=0.02):
        self.limit = int(limit_gib * 1024 ** 3)
        self.interval = interval
        self.peak = 0
        self.start = None
        self.elapsed = None
        self.stop_event = threading.Event()
        self.exceeded = False

    def _sample(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            rss = process.memory_info().rss
            self.peak = max(self.peak, rss)
            self.exceeded = self.exceeded or rss > self.limit
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.start = time.perf_counter()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop_event.set()
        self.worker.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.exceeded = self.exceeded or self.peak > self.limit
        self.elapsed = time.perf_counter() - self.start
        if self.exceeded and kind is None:
            raise MemoryError("Sampled process memory exceeded the declared pilot limit")

    def record(self):
        return {
            "elapsed_seconds": self.elapsed,
            "sampled_process_peak_rss_bytes": self.peak,
            "sampling_interval_seconds": self.interval,
            "limit_bytes": self.limit,
            "limit_exceeded": self.exceeded,
            "scope": "Whole Python process including loaded data; sampled RSS, not an operating-system exact peak",
        }
