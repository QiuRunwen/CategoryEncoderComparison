"""
test models.py
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
import models


class TestModels(unittest.TestCase):
    """test models.py"""

    def setUp(self) -> None:
        self.X_train = pd.DataFrame({"a": np.random.rand(10), "b": np.random.rand(10)}).to_numpy()
        self.y_train_cls = pd.Series([1] * 5 + [0] * 5).astype("category").to_numpy()
        self.y_train_reg = pd.Series(np.random.rand(10)).to_numpy()

        self.X_test = pd.DataFrame({"a": np.random.rand(10), "b": np.random.rand(10)}).to_numpy()
        self.y_test_cls = pd.Series([1] * 5 + [0] * 5).astype("category").to_numpy()
        self.y_test_reg = pd.Series(np.random.rand(10)).to_numpy()

        return super().setUp()

    def test_get_model(self):
        """test models.get_model function"""
        # with open(
        #     os.path.join(root_dir, "exp_conf", "exp_test.json"),
        #     mode="r",
        #     encoding="utf-8",
        # ) as file:
        #     exp_conf = json.load(file)

        # dict_model_kwargs = exp_conf["model_func_kwargs"]

        dict_model_kwargs = models.get_default_name_kwargs()
        for model_name, kwargs in dict_model_kwargs.items():
            classifier_names = models.get_available_model_name(model_type="classifier")
            regressor_names = models.get_available_model_name(model_type="regressor")
            if model_name in classifier_names:
                model, time_cost = models.train_model(
                    self.X_train, self.y_train_cls, model_name, **kwargs["kwargs"]
                )
                y_pred = model.predict(self.X_test)
                # y_pred_proba = model.predict_proba(self.X_test)
            elif model_name in regressor_names:
                model,time_cost = models.train_model(
                    self.X_train, self.y_train_reg, model_name, **kwargs["kwargs"]
                )
                y_pred = model.predict(self.X_test)
            else:
                raise ValueError(f"model_name: {model_name} is not supported.")


if __name__ == "__main__":
    unittest.main()
