"""
test datasets.py
"""
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import datasets


class TestDatasets(unittest.TestCase):
    """test datasets.py"""

    def test_load_one_set(self):
        """test datasets.load_one_set function"""
        # with open(
        #     os.path.join(root_dir, "exp_conf", "exp_test.json"),
        #     mode="r",
        #     encoding="utf-8",
        # ) as file:
        #     exp_conf = json.load(file)

        # dict_dataset_kwargs = exp_conf["dataset_func_kwargs"]
        dict_dataset_kwargs = {
            "Wholesale": {"func_name": "Wholesale", "kwargs": {}},
            "Cholesterol": {"func_name": "Cholesterol", "kwargs": {}},
        }

        for dataset_name, kwargs in dict_dataset_kwargs.items():
            dataset = datasets.load_one_set(
                kwargs["func_name"], dataset_name, **kwargs["kwargs"]
            )
            self.assertIsInstance(dataset, datasets.Dataset)

    def test_load_all_sets(self):
        """test datasets.load_all_sets function"""

        data_loader_dict = datasets.get_data_loader()
        for dataset in datasets.load_all(data_loader_dict=data_loader_dict):
            # binary classification task only contains 2 classes [0, 1] not [-1, 1]
            if not dataset.is_task_reg:
                tmpclss = np.unique(dataset.y)
                self.assertEqual(len(tmpclss), 2)
                self.assertTrue(0 in tmpclss and 1 in tmpclss)
            
            self.assertIsInstance(dataset, datasets.Dataset)

    def test_calc_col_card(self):
        df = pd.DataFrame(
            {
                "a": np.arange(10) + 0.1,
                "b": [1] * 10,
                "c": pd.Series([f"s{i}" for i in range(10)]).astype("category"),
                "d": pd.Series([1, 2, 3, 4] + [np.nan] * 6).astype("category"),
                "y": pd.Series([1] * 4 + [0] * 6).astype("category"),
            }
        )

        self.assertEqual(datasets.calc_col_card(df, "y").sum(), 10 + 5)
        self.assertEqual(datasets.calc_col_card(df).sum(), 10 + 5 + 2)

        self.assertEqual(
            datasets.calc_col_card(df, "y").fillna(1).sum(), 1 + 1 + 10 + 5
        )
        self.assertEqual(datasets.calc_col_card(df).fillna(1).sum(), 1 + 1 + 10 + 5 + 2)


if __name__ == "__main__":
    unittest.main()
