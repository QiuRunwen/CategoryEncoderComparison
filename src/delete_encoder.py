"""Delete encoder.

This module contains the :class:`DeleteEncoder` class, which is used to delete specified categorical columns.
"""
import warnings
import unittest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DeleteEncoder(BaseEstimator):
    """Delete specified categorical columns.

    Parameters
    ----------
    cols: list of str
        The columns to be deleted.
    """

    def __init__(self, delete_type="all"):
        self.delete_type = delete_type
        self.cols = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the DeleteEncoder.

        Parameters
        ----------
        X: pandas DataFrame
            The input data of shape (n_samples, n_features).
        y: None
            Ignored.

        Returns
        -------
        self: DeleteEncoder
            The fitted DeleteEncoder.
        """
        cat_cols = X.select_dtypes(include=["category"]).columns
        if self.delete_type == "highest_card":
            col = X[cat_cols].nunique().sort_values(ascending=False).index[0]
            self.cols = [col]
        elif self.delete_type == "all":
            self.cols = cat_cols
        else:
            raise ValueError(f"delete_type {self.delete_type} is not supported.")

        return self

    def transform(self, X: pd.DataFrame):
        """Transform the input data.

        Parameters
        ----------
        X: pandas DataFrame
            The input data of shape (n_samples, n_features).

        Returns
        -------
        X_transformed: pandas DataFrame
            The transformed data of shape (n_samples, n_features - len(cols)).
        """
        X_transformed = X.drop(columns=self.cols)
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform the input data.

        Parameters
        ----------
        X: pandas DataFrame
            The input data of shape (n_samples, n_features).
        y: None
            Ignored.

        Returns
        -------
        X_transformed: pandas DataFrame
            The transformed data of shape (n_samples, n_features - len(cols)).
        """
        return self.fit(X, y).transform(X)


class TestDeleteEncoder(unittest.TestCase):
    def test_delete_encoder(self):
        df = pd.DataFrame(
            {
                "x1": [1, 2, 3],
                "x2": ["4", "5", "4"],
                "x3": ["a", "a", "a"],
                "y": [1, 0, 1],
            }
        )

        df["x2"] = df["x2"].astype("category")
        df["x3"] = df["x3"].astype("category")

        X = df.drop(columns=["y"])
        y = df["y"]

        encoder = DeleteEncoder(delete_type="highest_card")
        X_transformed = encoder.fit_transform(X)
        self.assertListEqual(X_transformed.columns.tolist(), ["x1", "x3"])

        encoder = DeleteEncoder(delete_type="all")
        X_transformed = encoder.fit_transform(X)
        self.assertListEqual(X_transformed.columns.tolist(), ["x1"])

        X["x1"] = X["x1"].astype("category")
        encoder = DeleteEncoder(delete_type="all")
        X_transformed = encoder.fit_transform(X)
        self.assertListEqual(X_transformed.columns.tolist(), [])


if __name__ == "__main__":
    unittest.main()
