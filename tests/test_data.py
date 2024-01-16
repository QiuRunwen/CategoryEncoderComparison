import os
import sys
import unittest

import pandas as pd
import numpy as np

from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(Path(__file__).parent.parent, "src")))

from data.util import find_useless_colum, drop_useless, sampling_by_class


class TestUtil(unittest.TestCase):
    def test_find_useless_colum(self):
        df = pd.DataFrame(
            {
                "normal": np.arange(10) + 0.1,
                "a": [np.nan] * 10,
                "a1": [None] * 10,
                "b": [1] * 10,
                "b1": ["1s"] * 10,
                "c": [f"s{i}" for i in range(10)],
                "d": pd.Series([1, 1, 1, 5] + [np.nan] * 6).astype("category"),
                "e": [f"s{i}" for i in range(9)] + ["s1"],
                "f": ["s1"] + ["s2"] * 9,
            }
        )
        d_type_ls = find_useless_colum(
            df,
            max_missing_ratio=0.5,
            min_rows_per_value=2,
            max_ratio_per_cat=0.8,
            verbose=True,
        )

        self.assertDictEqual(
            {
                "empty_cols": ["a", "a1"],
                "single_value_cols": ["b", "b1"],
                "id_like_cols": ["c"],
                "too_many_missing_cols": ["d"],
                "too_small_cat_cols": ["e"],
                "too_large_cat_cols": ["f"],
            },
            d_type_ls,
        )

    def test_drop_useless(self):
        df = pd.DataFrame(
            {
                "normal": np.arange(10) + 0.1,
                "a": [np.nan] * 10,
                "a1": [None] * 10,
                "b": [1] * 10,
                "b1": ["1s"] * 10,
                "c": [f"s{i}" for i in range(10)],
                "d": pd.Series([1, 1, 1, 5] + [np.nan] * 6).astype("category"),
                "e": [f"s{i}" for i in range(9)] + ["s1"],
                "f": ["s1"] + ["s2"] * 9,
            }
        )
        d_type_ls = {
            "empty_cols": ["a", "a1"],
            "single_value_cols": ["b", "b1"],
            "id_like_cols": ["c"],
            "too_many_missing_cols": ["d"],
            "too_small_cat_cols": ["e"],
            "too_large_cat_cols": ["f"],
        }
        df_new = drop_useless(df, d_type_ls, verbose=True)
        self.assertEqual(df_new.columns.size, 1)
        self.assertEqual(df_new.columns[0], "normal")

    # def test_is_one_to_one(self):
    #     df = pd.DataFrame({
    #         'normal':[1.1,1.2,1.3],
    #         'a': [1,2,3],
    #         'b': ['a','b','c'],
    #     })
    #     ser = is_one_to_one(df, 'a', 'b')

    def test_sampling_by_class(self):
        df = pd.DataFrame(
            {
                "normal": np.arange(10) + 0.1,
                "a": pd.Series([1] * 4 + [0] * 6).astype("category"),
            }
        )
        df_new = sampling_by_class(df, class_col="a", num_sample=5)
        ser = df_new["a"].value_counts()
        # print(ser)
        self.assertEqual(ser.loc[1], 2)
        self.assertEqual(ser.loc[0], 3)

   


if __name__ == "__main__":
    unittest.main()
