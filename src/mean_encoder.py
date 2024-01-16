r"""Mean Encoder

Mean Encoder is simply modified from `category_encoders\target_encoder.py`
"""
import numpy as np
import pandas as pd

# from scipy.special import expit
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

from utils import format_y_binary

__author__ = "RunwenQiu"  # target_encoder的原作者是'chappers'


class MeanEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Mean encoder for categorical features."""

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_missing="value",
        handle_unknown="value",
        min_samples_leaf=20,
        smoothing=10,
        hierarchy=None,
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.mapping = None
        self._mean = None
        if isinstance(hierarchy, (dict, pd.DataFrame)) and cols is None:
            raise ValueError(
                "Hierarchy is defined but no columns are named for encoding"
            )
        if isinstance(hierarchy, dict):
            self.hierarchy = {}
            self.hierarchy_depth = {}
            for switch in hierarchy:
                flattened_hierarchy = util.flatten_reverse_dict(hierarchy[switch])
                hierarchy_check = self._check_dict_key_tuples(flattened_hierarchy)
                self.hierarchy_depth[switch] = hierarchy_check[1]
                if not hierarchy_check[0]:
                    raise ValueError(
                        'Hierarchy mapping contains different levels for key "'
                        + switch
                        + '"'
                    )
                self.hierarchy[switch] = {
                    (k if isinstance(t, tuple) else t): v
                    for t, v in flattened_hierarchy.items()
                    for k in t
                }
        elif isinstance(hierarchy, pd.DataFrame):
            self.hierarchy = hierarchy
            self.hierarchy_depth = {}
            for col in self.cols:
                HIER_cols = self.hierarchy.columns[
                    self.hierarchy.columns.str.startswith(f"HIER_{col}")
                ].values
                HIER_levels = [int(i.replace(f"HIER_{col}_", "")) for i in HIER_cols]
                if np.array_equal(
                    sorted(HIER_levels), np.arange(1, max(HIER_levels) + 1)
                ):
                    self.hierarchy_depth[col] = max(HIER_levels)
                else:
                    raise ValueError(
                        f"Hierarchy columns are not complete for column {col}"
                    )
        elif hierarchy is None:
            self.hierarchy = hierarchy
        else:
            raise ValueError(
                "Given hierarchy mapping is neither a dictionary nor a dataframe"
            )

        self.cols_hier = []

    def _check_dict_key_tuples(self, d):
        min_tuple_size = min(len(v) for v in d.values())
        max_tuple_size = max(len(v) for v in d.values())
        return min_tuple_size == max_tuple_size, min_tuple_size

    def _fit(self, X, y, **kwargs):
        # check if y is binary. modified by RunwenQiu
        if len(np.unique(y)) == 2:
            y = pd.Series(
                format_y_binary(y, neg_and_pos=False), index=X.index, name="y"
            )

        if isinstance(self.hierarchy, dict):
            X_hier = pd.DataFrame()
            for switch in self.hierarchy:
                if switch in self.cols:
                    colnames = [
                        f"HIER_{str(switch)}_{str(i + 1)}"
                        for i in range(self.hierarchy_depth[switch])
                    ]
                    df = pd.DataFrame(
                        X[str(switch)].map(self.hierarchy[str(switch)]).tolist(),
                        index=X.index,
                        columns=colnames,
                    )
                    X_hier = pd.concat([X_hier, df], axis=1)
        elif isinstance(self.hierarchy, pd.DataFrame):
            X_hier = self.hierarchy

        if isinstance(self.hierarchy, (dict, pd.DataFrame)):
            enc_hier = OrdinalEncoder(
                verbose=self.verbose,
                cols=X_hier.columns,
                handle_unknown="value",
                handle_missing="value",
            )
            enc_hier = enc_hier.fit(X_hier)
            X_hier_ordinal = enc_hier.transform(X_hier)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown="value",
            handle_missing="value",
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        if self.hierarchy is not None:
            self.mapping = self.fit_target_encoding(
                pd.concat([X_ordinal, X_hier_ordinal], axis=1), y
            )
        else:
            self.mapping = self.fit_target_encoding(X_ordinal, y)

    def fit_target_encoding(self, X, y):
        mapping = {}

        prior = self._mean = y.mean()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get("col")
            if "HIER_" not in str(col):
                values = switch.get("mapping")

                scalar = prior
                if (isinstance(self.hierarchy, dict) and col in self.hierarchy) or (
                    isinstance(self.hierarchy, pd.DataFrame)
                ):
                    for i in range(self.hierarchy_depth[col]):
                        col_hier = "HIER_" + str(col) + "_" + str(i + 1)
                        col_hier_m1 = (
                            col
                            if i == self.hierarchy_depth[col] - 1
                            else "HIER_" + str(col) + "_" + str(i + 2)
                        )
                        if (
                            not X[col].equals(X[col_hier])
                            and len(X[col_hier].unique()) > 1
                        ):
                            stats_hier = y.groupby(X[col_hier]).agg(["count", "mean"])
                            smoove_hier = self._weighting(stats_hier["count"])
                            scalar_hier = (
                                scalar * (1 - smoove_hier)
                                + stats_hier["mean"] * smoove_hier
                            )
                            scalar_hier_long = X[
                                [col_hier_m1, col_hier]
                            ].drop_duplicates()
                            scalar_hier_long.index = np.arange(
                                1, scalar_hier_long.shape[0] + 1
                            )
                            scalar = scalar_hier_long[col_hier].map(
                                scalar_hier.to_dict()
                            )

                stats = y.groupby(X[col]).agg(["count", "mean"])
                smoove = self._weighting(stats["count"])

                smoothing = scalar * (1 - smoove) + stats["mean"] * smoove

                if self.handle_unknown == "return_nan":
                    smoothing.loc[-1] = np.nan
                elif self.handle_unknown == "value":
                    smoothing.loc[-1] = prior

                if self.handle_missing == "return_nan":
                    smoothing.loc[values.loc[np.nan]] = np.nan
                elif self.handle_missing == "value":
                    smoothing.loc[-2] = prior

                mapping[col] = smoothing

        return mapping

    def _transform(self, X, y=None):
        # Now X is the correct dimensions it works with pre fitted ordinal encoder
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == "error":
            if X[self.cols].isin([-1]).any().any():
                raise ValueError("Unexpected categories found in dataframe")

        X = self.target_encode(X)
        return X

    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        # Was not mapping extra columns as self.featuer_names_in did not include new column
        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X

    def _weighting(self, n):
        # monotonically increasing function of n bounded between 0 and 1
        # sigmoid in this case, using scipy.expit for numerical stability
        # return expit((n - self.min_samples_leaf) / self.smoothing)
        return 1  # modified by RunwenQiu
