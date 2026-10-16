from pathlib import Path

import numpy as np

from .records import array_hash, file_hash


def assert_separated(first, second, groups):
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if len(np.unique(first)) != len(first) or len(np.unique(second)) != len(second):
        raise ValueError("Repeated row position within a partition")
    if np.intersect1d(first, second).size:
        raise ValueError("Row overlap across partitions")
    if np.intersect1d(groups[first], groups[second]).size:
        raise ValueError("Corrected-vector group overlap across partitions")


def load_verified_splits(project_root, protocol, groups):
    path = Path(project_root) / protocol["splits"]["source"]
    if file_hash(path) != protocol["splits"]["artifact_sha256"]:
        raise ValueError("Historical split artifact identity differs")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if not np.array_equal(groups, arrays["corrected_group"]):
        raise ValueError("Corrected group identities differ from verified historical groups")
    mapping = arrays["outer_fold"]
    all_rows = np.arange(len(groups))
    if mapping.shape != all_rows.shape or not np.isin(mapping, np.arange(1, 6)).all():
        raise ValueError("Outer mapping does not cover the fixed rows")
    for record in protocol["splits"]["fold_index_hashes"]:
        fold = record["fold"]
        part = {name: arrays[f"fold_{fold}_{name}"] for name in record if name != "fold"}
        for name, rows in part.items():
            if array_hash(rows) != record[name]:
                raise ValueError(f"Ordered partition identity differs: {fold}, {name}")
            if np.any(rows < 0) or np.any(rows >= len(groups)):
                raise ValueError("Partition row is out of range")
        assert_separated(part["outer_train"], part["outer_test"], groups)
        assert_separated(part["inner_train"], part["inner_validation"], groups)
        if not np.array_equal(part["outer_test"], np.flatnonzero(mapping == fold)):
            raise ValueError("Outer rows differ from historical membership")
        if not np.array_equal(np.sort(np.concatenate([part["outer_train"], part["outer_test"]])), all_rows):
            raise ValueError("Incomplete outer partition")
        if not np.array_equal(np.sort(np.concatenate([part["inner_train"], part["inner_validation"]])), part["outer_train"]):
            raise ValueError("Inner partition differs from outer-training membership")
    return arrays
