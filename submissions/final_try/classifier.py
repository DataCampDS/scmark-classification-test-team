import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif


# 1. Custom Preprocessing
def rna_log_norm(X):
    # RAMP gives us a sparse matrix, so we convert it to a normal array first
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Calculate library size (sum of row)
    library_size = X.sum(axis=1, keepdims=True)
    # Fix rows with 0 sum so we don't divide by zero
    library_size[library_size == 0] = 1

    # Normalize and Log Transform
    return np.log1p((X / library_size) * 10000)


# 2. Custom Feature Selector
class FocusedFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k_general=2000, k_specific=150):
        self.k_general = k_general
        self.k_specific = k_specific
        self.specific_classes = ['NK_cells', 'T_cells_CD8+', 'T_cells_CD4+']
        self.selected_indices_ = None

    def fit(self, X, y=None):
        # Step 1: General Feature Selection
        # Use ANOVA (f_classif) to find genes that separate all classes well
        if y is not None:
            f_scores, _ = f_classif(X, y)
            f_scores = np.nan_to_num(f_scores)  # Fix errors
            ind_general = np.argsort(f_scores)[-self.k_general:]
        else:
            # If no labels, just use variance
            vars_ = np.var(X, axis=0)
            ind_general = np.argsort(vars_)[-self.k_general:]

        # Step 2: Specific Immune Selection
        # Look for genes that help distinguish the confusing immune cells
        ind_specific = np.array([], dtype=int)

        if y is not None:
            y_arr = np.array(y)
            # Filter for just the immune classes we care about
            mask = np.isin(y_arr, self.specific_classes)

            if mask.sum() > 0:
                X_sub = X[mask]
                y_sub = y_arr[mask]

                # We need at least 2 different classes to compare them
                if len(np.unique(y_sub)) >= 2:
                    f_scores_spec, _ = f_classif(X_sub, y_sub)
                    f_scores_spec = np.nan_to_num(f_scores_spec)
                    ind_specific = np.argsort(f_scores_spec)[-self.k_specific:]

        # Step 3: Combine both lists
        self.selected_indices_ = np.union1d(ind_general, ind_specific)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]


# 3. The RAMP Classifier
class Classifier(object):
    def __init__(self):
        # We build the pipeline here with the settings we found earlier
        self.pipe = make_pipeline(
            FunctionTransformer(rna_log_norm),
            FocusedFeatureSelector(k_general=2000, k_specific=150),
            PCA(n_components=30, random_state=42),
            SVC(
                C=1,
                gamma=0.001,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        )

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)