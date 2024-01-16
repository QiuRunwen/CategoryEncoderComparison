import os
import models
import preprocess
import datasets

import utils

def get_default_conf(save_path=None):
    res = {
        "dataset_func_kwargs": datasets.get_default_name_kwargs(sort=True),
        "encoder_func_kwargs": preprocess.get_default_name_kwargs(),
        "model_func_kwargs": models.get_default_name_kwargs(),
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": "../output/result/test",
            "test_size": 0.2,
            "use_cache": False,
        },
    }

    res["model_func_kwargs"]["DecisionTreeClassifier"] = {
        "func_name": "DecisionTreeClassifier",
        "kwargs": {"max_depth": 10, "min_samples_leaf": 10},
    }
    res["model_func_kwargs"]["DecisionTreeRegressor"] = {
        "func_name": "DecisionTreeRegressor",
        "kwargs": {"max_depth": 10, "min_samples_leaf": 10},
    }
    res["model_func_kwargs"]["XGBClassifier"] = {
        "func_name": "XGBClassifier",
        "kwargs": {"tree_method": "hist", "enable_categorical": True},
    }
    res["model_func_kwargs"]["XGBRegressor"] = {
        "func_name": "XGBRegressor",
        "kwargs": {"tree_method": "hist", "enable_categorical": True},
    }

    res["encoder_func_kwargs"]["Auto"] = {"func_name": "Auto", "kwargs": {}}
    res["encoder_func_kwargs"]["LeaveOneOutEncoder"] = {
        "func_name": "LeaveOneOutEncoder",
        "kwargs": {"sigma": 0.05},
    }
    res["encoder_func_kwargs"]["DeleteEncoder"] = {
        "func_name": "DeleteEncoder",
        "kwargs": {"delete_type": "all"},
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        utils.save(res, save_path)
        print(f"Default configuration saved to {save_path}")

    return res



if __name__ == "__main__":
    get_default_conf("../exp_conf/default_conf.json")

