"""Experiment pipeline."""

import itertools
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from datasets import load_one_set
from models import train_model, get_available_model_name, ModelOnlyUseTarget, evaluate
from preprocess import split_impute_encode_scale, get_available_encoder_name


def check_exp_id_executed(
    exp_id: str,
    executed_exp_ids: list[str],
    *tup_upcoming_args: list[str],
    is_task_reg: bool = None,
    encoder_func_name: str = None,
    model_func_names: list[str] = None
) -> bool:
    """Check if the experiment has been executed.

    Args:
        exp_id (str): Current experiment id. e.g. "0-Adult-TargetEncoder"
        executed_exp_ids (list[str]): List of executed experiment ids. e.g. ["0-Adult-TargetEncoder-LogisticRegressionCV"]
        *tup_upcoming_args (list[str]): List of upcoming arguments. e.g. ["LogisticRegressionCV"]

    Returns:
        bool: _description_
    """
    if tup_upcoming_args:
        upcoming_exp_ids = [
            "-".join(arg_tuple)
            for arg_tuple in itertools.product([exp_id], *tup_upcoming_args)
        ]
    else:
        upcoming_exp_ids = [exp_id]

    if is_task_reg is not None:
        ser_ids = pd.Series(upcoming_exp_ids)
        # df = ser_ids.str.split("-", expand=True)
        # df.columns = 0,1,2,3. ["seed", "dataset", "encoder", "model"]
        # df.columns = ["seed", "dataset", "encoder", "model"]
        assert ser_ids.shape[0] == len(model_func_names)
        encoder_func_names = [encoder_func_name] * len(model_func_names)

        def check(encoder_func_name, model_func_name):
            return check_task_encoder_model(
                is_task_reg, encoder_func_name, model_func_name
            )

        tmp_filter = np.vectorize(check)(encoder_func_names, model_func_names)
        upcoming_exp_ids = ser_ids[tmp_filter].tolist()

    # `all(e in ls2 for e in ls1)`, a generator expression is used to create an iterator that yields True or False.
    # `all([e in ls2 for e in ls1])`, a list comprehension is used instead of a generator expression
    #  first encoder using a generator expression is more memory-efficient
    return all(
        upcoming_exp_id in executed_exp_ids for upcoming_exp_id in upcoming_exp_ids
    )


def check_task_encoder_model(
    is_task_reg: bool, encoder_func_name: str = None, model_func_name: str = None
) -> bool:
    if encoder_func_name == "Auto":
        if model_func_name is not None:
            if is_task_reg:
                return model_func_name in [
                    "CatBoostRegressor",
                    "XGBRegressor",
                    "LGBMRegressor",
                ]

            else:
                return model_func_name in [
                    "CatBoostClassifier",
                    "XGBClassifier",
                    "LGBMClassifier",
                ]

    is_ok = False
    if encoder_func_name is not None:
        target_type = "continuous" if is_task_reg else "binomial"
        enocoder_names = get_available_encoder_name(target_type)
        enocoder_names.append("Auto")
        is_ok = encoder_func_name in enocoder_names

    if model_func_name is not None:
        model_type = "regressor" if is_task_reg else "classifier"
        models = get_available_model_name(model_type)

        is_ok = model_func_name in models

    return is_ok


class ObjHasShape:
    __slots__ = ["shape"]

    def __init__(self, shape):
        self.shape = shape


def exp(
    dataset_func_kwargs: dict[str, dict],
    encoder_func_kwargs: dict[str, dict],
    model_func_kwargs: dict[str, dict],
    test_size: float = 0.2,
    seeds: list[int] = None,
    output_dir: str = "./output",
    use_cache: bool = False,
    delete_highest: bool = False,
):
    """experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logging.captureWarnings(True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "exp.log"),
        level=logging.INFO,
        format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
        encoding="utf-8",
    )
    logging.info("------start exp------")
    if seeds is None:
        warnings.warn("seeds is None, will use a random int as seed")
        seeds = [np.random.randint(0, 10000)]

    result_path = os.path.join(output_dir, "result.csv")
    df_result = (
        pd.read_csv(result_path) if os.path.exists(result_path) else pd.DataFrame()
    )

    executed_exp_ids = (
        df_result["exp_id"].unique().tolist() if "exp_id" in df_result.columns else []
    )

    result_dicts = []
    for seed in tqdm(seeds, desc="seed"):
        if seed is None:
            seed = np.random.randint(0, 10000)
            utils.seed_everything(seed)

        exp_id_seed = str(seed)
        for dataset_name, tmp_dict in tqdm(
            dataset_func_kwargs.items(), desc=exp_id_seed
        ):
            exp_id_seed_dataset = exp_id_seed + "-" + dataset_name
            if check_exp_id_executed(
                exp_id_seed_dataset,
                executed_exp_ids,
                encoder_func_kwargs.keys(),
                model_func_kwargs.keys(),
            ):
                logging.info("exp_id: %s has been executed, skip", exp_id_seed_dataset)
                continue

            dataset_func_name = tmp_dict["func_name"]
            dataset_kwargs = tmp_dict["kwargs"]

            utils.seed_everything(seed)
            dataset = load_one_set(
                func_name=dataset_func_name, name=dataset_name, **dataset_kwargs
            )

            for encoder_name, tmp_kwargs in tqdm(
                encoder_func_kwargs.items(), desc=exp_id_seed_dataset
            ):
                if not check_task_encoder_model(
                    dataset.is_task_reg, tmp_kwargs["func_name"], None
                ):
                    logging.info(
                        "encoder: %s is not suitable for task: %s, skip",
                        tmp_kwargs["func_name"],
                        dataset.name,
                    )
                    continue

                exp_id_seed_dataset_encoder = exp_id_seed_dataset + "-" + encoder_name
                if check_exp_id_executed(
                    exp_id_seed_dataset_encoder,
                    executed_exp_ids,
                    model_func_kwargs.keys(),
                    is_task_reg=dataset.is_task_reg,
                    model_func_names=[
                        tmp_kwargs["func_name"]
                        for _, tmp_kwargs in model_func_kwargs.items()
                    ],
                ):
                    logging.info(
                        "exp_id: %s has been executed, skip",
                        exp_id_seed_dataset_encoder,
                    )
                    continue

                encoder_func_name = tmp_kwargs["func_name"]
                encoder_kwargs = tmp_kwargs["kwargs"]
                # encoder_cache_dir = None
                # if use_cache:
                #     encoder_cache_dir = os.path.join(
                #         output_dir,
                #         "cache",
                #         exp_id_seed_dataset_encoder.replace("-", "/"),
                #     )

                utils.seed_everything(seed)  # splitting and encoding need seed
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    time_encoder,
                ) = split_impute_encode_scale(
                    dataset=dataset,
                    test_size=test_size,
                    encoder_func_name=encoder_func_name,
                    encoder_kwargs=encoder_kwargs,
                    seed=seed,
                    # encoder_cache_dir=encoder_cache_dir, # TODO
                    delete_highest=delete_highest,
                )

                if X_train is None or len(X_train)==0:
                    # some dataset only have categorical features
                    # when deleting categorical features, the dataset will be empty
                    # in this case, we can only use the target to train a model
                    X_train = ObjHasShape((y_train.shape[0], 0))
                    X_test = ObjHasShape((y_test.shape[0], 0))

                for model_name, tmp_kwargs in tqdm(
                    model_func_kwargs.items(), desc=exp_id_seed_dataset_encoder
                ):
                    if not check_task_encoder_model(
                        dataset.is_task_reg, encoder_func_name, tmp_kwargs["func_name"]
                    ):
                        logging.info(
                            "model: %s is not suitable for task: %s or encoder: %s, skip",
                            tmp_kwargs["func_name"],
                            "reg" if dataset.is_task_reg else "clf",
                            encoder_func_name,
                        )
                        continue

                    exp_id_seed_dataset_encoder_model = (
                        exp_id_seed_dataset_encoder + "-" + model_name
                    )
                    if check_exp_id_executed(
                        exp_id_seed_dataset_encoder_model, executed_exp_ids
                    ):
                        logging.info(
                            "exp_id: %s has been executed, skip",
                            exp_id_seed_dataset_encoder_model,
                        )
                        continue

                    model_func_name = tmp_kwargs["func_name"]
                    model_kwargs = tmp_kwargs["kwargs"]
                    model_cache_dir = None
                    if use_cache:
                        model_cache_dir = os.path.join(
                            output_dir,
                            "cache",
                            exp_id_seed_dataset_encoder_model.replace("-", "/"),
                        )

                    utils.seed_everything(seed)

                    if isinstance(X_train, ObjHasShape):
                        model = ModelOnlyUseTarget(
                            func_name=model_func_name,
                            kwargs=model_kwargs,
                            is_task_reg=dataset.is_task_reg,
                        )
                        model.fit(X_train, y_train)
                        time_model = 0
                    else:
                        model, time_model = train_model(
                            func_name=model_func_name,
                            X_train=X_train,
                            y_train=y_train,
                            kwargs=model_kwargs,
                            cache_dir=model_cache_dir,
                        )
                    dict_res = {
                        "exp_id": exp_id_seed_dataset_encoder_model,
                        "seed": seed,
                        "dataset": dataset_name,
                        "is_task_reg": dataset.is_task_reg,
                        "train_size": X_train.shape[0],
                        "test_size": X_test.shape[0],
                        "train_dim": X_train.shape[1],
                        "encoder": encoder_name,
                        "time_encoder": time_encoder,
                        "model": model_name,
                        "time_model": time_model,
                    }

                    for X_to_eval, y_to_eval, suffix in [
                        (X_test, y_test, ""),
                        (X_train, y_train, "_train"),
                    ]:
                        dict_res.update(
                            evaluate(
                                model=model,
                                is_task_reg=dataset.is_task_reg,
                                X=X_to_eval,
                                y=y_to_eval,
                                suffix=suffix,
                            )
                        )

                    result_dicts.append(dict_res)

                    utils.append_dicts_to_csv(result_path, [dict_res])
                    executed_exp_ids.append(exp_id_seed_dataset_encoder_model)

    df_result = pd.concat([df_result, pd.DataFrame(result_dicts)], axis=0)

    return df_result


def _test():
    dict_dataset_func_kwargs = {
        "Adult": {
            "func_name": "Adult",
            "kwargs": {},
        },
        "Colleges": {
            "func_name": "Colleges",
            "kwargs": {},
        },
    }

    dict_encoder_func_kwargs = {
        "Target": {
            "func_name": "TargetEncoder",
            "kwargs": {},
        },
        "LeaveOneOut": {
            "func_name": "LeaveOneOutEncoder",
            "kwargs": {},
        },
        "WOE": {
            "func_name": "WOEEncoder",
            "kwargs": {},
        },
    }

    dict_model_func_kwargs = {
        "LogisticRegressionCV": {
            "func_name": "LogisticRegressionCV",
            "kwargs": {},
        },
        "RidgeCV": {
            "func_name": "RidgeCV",
            "kwargs": {},
        },
    }

    other_kwargs = {
        "seeds": [0, 1, 2, 3, 4],
        "output_dir": "../output/result/test",
        "test_size": 0.2,
        "use_cache": False,
    }

    exp(
        dict_dataset_func_kwargs,
        dict_encoder_func_kwargs,
        dict_model_func_kwargs,
        **other_kwargs
    )


def _test2():
    exp_dict = utils.load("../exp_conf/exp_conf.json")
    utils.save(
        exp_dict, os.path.join(exp_dict["other_kwargs"]["output_dir"], "conf.json")
    )
    exp(
        dataset_func_kwargs=exp_dict["dataset_func_kwargs"],
        encoder_func_kwargs=exp_dict["encoder_func_kwargs"],
        model_func_kwargs=exp_dict["model_func_kwargs"],
        **exp_dict["other_kwargs"]
    )


def main(exp_file_path=None):
    if exp_file_path is None and len(sys.argv) < 2:
        _test()
    else:
        exp_file_path = sys.argv[1] if exp_file_path is None else exp_file_path
        exp_dict = utils.load(exp_file_path)
        utils.save(
            exp_dict, os.path.join(exp_dict["other_kwargs"]["output_dir"], "conf.json")
        )
        exp(
            dataset_func_kwargs=exp_dict["dataset_func_kwargs"],
            encoder_func_kwargs=exp_dict["encoder_func_kwargs"],
            model_func_kwargs=exp_dict["model_func_kwargs"],
            **exp_dict["other_kwargs"]
        )


if __name__ == "__main__":
    main(
        "../exp_conf/exp_conf.json"
        # "../exp_conf/exp_conf_glmm.json"
        # "../exp_conf/var_imp_conf.json"
        # "../exp_conf/mlp_grad_add_col_conf.json"
        # "../exp_conf/mlp_grad_add_noise_col_conf.json"
        # "../exp_conf/mlp_grad_add_noise_catcol_conf.json"
    )
