"""
test exp.py
"""
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import exp
import exp_conf


class TestExp(unittest.TestCase):
    """test exp.py"""

    def test_exp(self):
        """test exp.exp function"""
        with open(
            os.path.join(root_dir, "exp_conf", "exp_test.json"),
            mode="r",
            encoding="utf-8",
        ) as file:
            conf = json.load(file)

        dict_dataset_kwargs = conf["dataset_func_kwargs"]
        dict_method_kwargs = conf["encoder_func_kwargs"]
        dict_model_kwargs = conf["model_func_kwargs"]
        other_kwargs = conf["other_kwargs"]
        other_kwargs["use_cache"] = False

        # remove the test dir and all its contents for a clean test
        if os.path.exists(other_kwargs["output_dir"]):
            shutil.rmtree(other_kwargs["output_dir"])

        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )
        other_kwargs["seeds"] = other_kwargs["seeds"] + [42]
        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )

    def test_default_conf(self):
        """test exp.exp function using default conf"""
        conf = exp_conf.get_default_conf()
        # dict_dataset_kwargs = conf["dataset_func_kwargs"]
        dict_dataset_kwargs = {
            "Wholesale": {"func_name": "Wholesale", "kwargs": {}},
            "Cholesterol": {"func_name": "Cholesterol", "kwargs": {}},
        }

        dict_method_kwargs = conf["encoder_func_kwargs"]
        dict_model_kwargs = conf["model_func_kwargs"]
        other_kwargs = conf["other_kwargs"]

        other_kwargs["output_dir"] = os.path.join(root_dir, "output/test_default_conf")
        other_kwargs["use_cache"] = False
        other_kwargs["seeds"] = [0, 1]

        # remove the test dir and all its contents for a clean test
        if os.path.exists(other_kwargs["output_dir"]):
            shutil.rmtree(other_kwargs["output_dir"])

        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )


if __name__ == "__main__":
    unittest.main()
