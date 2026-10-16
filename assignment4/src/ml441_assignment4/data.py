from pathlib import Path

import numpy as np
import pandas as pd

from .records import file_hash


def load_data(project_root, protocol):
    root = Path(project_root).resolve()
    spec = protocol["data"]
    paths = {
        "sha256": root / spec["file"],
        "features_sha256": root / "data/raw/features.csv",
        "class_map_sha256": root / "data/raw/attack_category_map.csv",
    }
    for key, path in paths.items():
        if file_hash(path) != spec[key]:
            raise ValueError(f"Data identity mismatch: {path.name}")
    mapping = pd.read_csv(paths["class_map_sha256"])
    if mapping["Mapping"].tolist() != spec["class_order"]:
        raise ValueError("Class order differs from the fixed map")
    if mapping["Attack Category Name"].tolist() != spec["class_names"]:
        raise ValueError("Class names differ from the fixed map")
    raw = pd.read_csv(paths["sha256"], na_values=["?"], low_memory=False)
    if raw.shape != (spec["rows"], 44) or raw.columns[-1] != spec["target"]:
        raise ValueError("Unexpected raw schema")
    target = pd.to_numeric(raw[spec["target"]], errors="raise").to_numpy(dtype=np.int8)
    if sorted(np.unique(target).tolist()) != spec["class_order"]:
        raise ValueError("Unexpected target labels")
    features = raw.drop(columns=[spec["target"], *protocol["representation"]["drop"]]).copy()
    nominal = protocol["representation"]["nominal"]
    for column in nominal:
        features[column] = features[column].astype("string").fillna("Missing")
    numeric = [column for column in features if column not in nominal]
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="raise")
    if not np.isfinite(features[numeric].to_numpy()).all():
        raise ValueError("Retained numerical predictors contain nonfinite values")
    return features, target


def corrected_groups(features):
    hashed = pd.util.hash_pandas_object(features, index=False, categorize=True).to_numpy()
    groups, _ = pd.factorize(hashed, sort=False)
    distinct = features.assign(group_identity=groups).drop_duplicates()
    if distinct["group_identity"].duplicated().any():
        raise ValueError("Unequal corrected vectors share a hash group")
    return groups.astype(np.int64, copy=False)
