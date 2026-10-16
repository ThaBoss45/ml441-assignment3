import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def make_preprocessor(features, nominal):
    numeric = [column for column in features if column not in nominal]
    return ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32, drop=None), nominal),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def forest_matrix(matrix, fitting=False):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.sort_indices()
    return matrix.tocsc() if fitting else matrix


def representation_record(preprocessor):
    encoder = preprocessor.named_transformers_["categorical"]
    return {
        "input_columns": preprocessor.feature_names_in_.tolist(),
        "output_columns": preprocessor.get_feature_names_out().tolist(),
        "categories": [values.tolist() for values in encoder.categories_],
        "numeric_columns": list(preprocessor.transformers_[0][2]),
        "nominal_columns": list(preprocessor.transformers_[1][2]),
        "unknown_category": "All-zero coordinates within that nominal block",
        "dtype": "float32",
        "fitting_scope": "Recorded fitting row positions only",
    }
