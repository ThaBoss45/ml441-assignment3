from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def make_forest(protocol, setting, seed, trees, workers=4, warm_start=False):
    return RandomForestClassifier(
        **protocol["ensemble"]["fixed"],
        **setting,
        n_estimators=trees,
        random_state=seed,
        n_jobs=workers,
        warm_start=warm_start,
    )


def probability_labels(probabilities, labels):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability columns do not match class labels")
    if not np.array_equal(labels, np.sort(np.unique(labels))):
        raise ValueError("Class labels must be sorted and unique for the declared tie rule")
    return labels[np.argmax(probabilities, axis=1)]


def member_probabilities(forest, tree, matrix, labels):
    labels = np.asarray(labels)
    local = tree.predict_proba(matrix)
    encoded = np.asarray(tree.classes_, dtype=int)
    member_labels = np.asarray(forest.classes_)[encoded]
    aligned = np.zeros((matrix.shape[0], len(labels)), dtype=np.float64)
    positions = {label: position for position, label in enumerate(labels)}
    for source, label in enumerate(member_labels):
        if label not in positions:
            raise ValueError("Member class lies outside the declared class order")
        aligned[:, positions[label]] = local[:, source]
    return aligned


def probability_aggregate(forest, matrix, labels, count=None):
    members = forest.estimators_[:count]
    if not members:
        raise ValueError("An ensemble needs at least one fitted member")
    total = np.zeros((matrix.shape[0], len(labels)), dtype=np.float64)
    hard = []
    for tree in members:
        probabilities = member_probabilities(forest, tree, matrix, labels)
        total += probabilities
        hard.append(probability_labels(probabilities, labels))
    return total / len(members), np.asarray(hard).T


def diversity_record(hard_predictions, labels):
    votes = np.stack([(hard_predictions == label).sum(axis=1) for label in labels], axis=1)
    members = hard_predictions.shape[1]
    pairs = members * (members - 1) // 2
    agree = (votes * (votes - 1)).sum(axis=1) / 2
    disagreement = None
    if members <= 16:
        disagreement = [float(np.mean(hard_predictions[:, first] != hard_predictions[:, second])) for first, second in combinations(range(members), 2)]
    hard_vote = probability_labels(votes, labels)
    return {
        "members": members,
        "pairs": pairs,
        "mean_pairwise_disagreement": float(np.mean(1 - agree / pairs)) if pairs else 0.0,
        "pairwise_disagreement": disagreement,
    }, hard_vote


def tree_costs(forest):
    return {
        "trees": len(forest.estimators_),
        "total_nodes": int(sum(tree.tree_.node_count for tree in forest.estimators_)),
        "max_depths": [int(tree.tree_.max_depth) for tree in forest.estimators_],
        "tree_seeds": [int(tree.random_state) for tree in forest.estimators_],
        "node_feature_limits": [int(tree.max_features_) for tree in forest.estimators_],
    }
