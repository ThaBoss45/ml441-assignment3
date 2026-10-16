import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import forest_matrix, make_preprocessor


def historical_preprocessor(features, nominal, neighbour=False):
    preprocessor = make_preprocessor(features, nominal)
    if neighbour:
        preprocessor.set_params(numeric=MinMaxScaler(clip=False))
    return preprocessor


def capped_positions(target, cap, seed, labels):
    rng = np.random.default_rng(seed)
    blocks = []
    for label in labels:
        indices = np.flatnonzero(target == label)
        if len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        blocks.append(np.sort(indices))
    result = np.concatenate(blocks)
    rng.shuffle(result)
    return result


def neighbour_vote(distances, indices, target, labels, setting):
    distances = distances[:, :setting["k"]]
    neighbours = target[indices[:, :setting["k"]]]
    if setting["weights"] == "uniform":
        weights = np.ones_like(distances, dtype=float)
    elif setting["weights"] == "distance":
        if setting["vote_exponent"] <= 0:
            raise ValueError("Inverse-distance exponent must be positive")
        zero = distances <= np.finfo(float).eps
        weights = np.divide(1, np.power(distances, setting["vote_exponent"]), out=np.zeros_like(distances, dtype=float), where=~zero)
        weights[zero.any(axis=1)] = zero[zero.any(axis=1)]
    else:
        raise ValueError("Unknown historical vote rule")
    scores = np.zeros((len(distances), len(labels)), dtype=np.float64)
    row_positions = np.arange(len(distances))
    encoded = np.searchsorted(labels, neighbours)
    for neighbour in range(setting["k"]):
        np.add.at(scores, (row_positions, encoded[:, neighbour]), weights[:, neighbour])
    return np.asarray(labels)[np.argmax(scores, axis=1)]


def fit_historical_tree(matrix, target, setting, fold):
    return DecisionTreeClassifier(**setting, random_state=20260826 + fold).fit(matrix, target)


def fit_historical_neighbours(matrix, target, setting, fold, labels):
    positions = capped_positions(target, 5000, 20260826 + fold + 200, labels)
    metric = "manhattan" if setting["p"] == 1 else "euclidean"
    estimator = NearestNeighbors(n_neighbors=setting["k"], metric=metric, algorithm="brute", n_jobs=4)
    estimator.fit(forest_matrix(matrix)[positions])
    return estimator, target[positions], positions
