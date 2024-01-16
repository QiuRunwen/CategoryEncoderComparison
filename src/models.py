""" This module contains the functions to train the models."""

import logging
import os
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
    LogisticRegression,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor


import utils
from datasets import get_num_cat_cols


CLASSIFIERS = (
    MLPClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    LogisticRegression,
    LogisticRegressionCV,
    GaussianNB,
    KNeighborsClassifier,
    SVC,
    LinearSVC,
    DecisionTreeClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
)

REGRESSORS = (
    MLPRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    LinearRegression,
    Ridge,
    RidgeCV,
    KNeighborsRegressor,
    SVR,
    LinearSVR,
    DecisionTreeRegressor,
    XGBRegressor,
    LGBMRegressor,
    CatBoostRegressor,
)

CLASSIFIER_NAMES = [func.__name__ for func in CLASSIFIERS]
REGRESSOR_NAMES = [func.__name__ for func in REGRESSORS]


class ModelOnlyUseTarget:
    def __init__(
        self, func_name: str = None, kwargs: dict = None, is_task_reg: bool = True
    ):
        self.func_name = func_name
        self.kwargs = kwargs if kwargs is not None else {}
        self.is_task_reg = is_task_reg

        self.y_mean = None
        self.classes = None
        self.num_classes = None
        self.classes_proba = None
        self.dominant_class = None

    def fit(self, X, y):
        if self.is_task_reg:
            self.y_mean = np.mean(y)
        else:
            y_unique, y_count = np.unique(y, return_counts=True)

            # sort by class to force the order of classes_
            # e.g. [1, 0] -> [0, 1]
            y_unique, y_count = zip(*sorted(zip(y_unique, y_count)))
            y_unique = np.array(y_unique)
            y_count = np.array(y_count)

            self.classes = y_unique
            self.num_classes = len(y_unique)
            self.classes_proba = y_count / y_count.sum()
            self.dominant_class = y_unique[np.argmax(y_count)]

    def predict(self, X):
        if self.is_task_reg:
            return np.full(X.shape[0], self.y_mean)
        else:
            return np.full(X.shape[0], self.dominant_class)

    def predict_proba(self, X):
        if self.is_task_reg:
            raise ValueError("predict_proba is not supported for regression.")
        else:
            return np.full((X.shape[0], self.num_classes), self.classes_proba)


D_FUNCNAME_FUNC = MappingProxyType(
    {func.__name__: func for func in CLASSIFIERS + REGRESSORS}
)


def get_default_name_kwargs() -> dict:
    res = {}
    for func_name in D_FUNCNAME_FUNC:
        res[func_name] = {
            "func_name": func_name,
            "kwargs": {},
        }
    return res


def get_available_model_name(model_type: str = None) -> list[str]:
    """Get the names of available models."""
    if model_type is None:
        return list(D_FUNCNAME_FUNC.keys())
    elif model_type == "classifier":
        return CLASSIFIER_NAMES.copy()
    elif model_type == "regressor":
        return REGRESSOR_NAMES.copy()
    else:
        raise ValueError(f"model_type: {model_type} is not supported.")


def _model_fit(model, X_train, y_train, seed=None):
    if type(model) in [SVC, SVR] and len(X_train) > 20000:
        # SVC and SVR are too slow, so we sample 20000 data points
        X_train, y_train = utils.random_sampling(
            X_train, y_train, sample_num=20000, stratify=True, seed=seed
        )

    if type(X_train) == pd.DataFrame:
        # category features don't have been encoded
        _, cat_cols, _, _ = get_num_cat_cols(X_train)
        if type(model) in [LGBMClassifier, LGBMRegressor]:
            model.fit(X_train, y_train, categorical_feature=cat_cols)
        elif type(model) in [CatBoostClassifier, CatBoostRegressor]:
            model.fit(X_train, y_train, cat_features=cat_cols)
        elif type(model) in [XGBClassifier, XGBRegressor]:
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"model: {model} is not supported for {X_train.shape}.")
    else:
        model.fit(X_train, y_train)


def train_model(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    func_name: str,
    kwargs: dict = None,
    cache_dir: str = None,
    seed: int = None,
) -> tuple[ClassifierMixin | RegressorMixin, float]:
    """Train a model with the given data and hyperparameters.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training label.
        func_name (str): The name of the model which is in `d_funcname_func`.
        kwargs (dict, optional): The hyperparameters of the model. Defaults to None.
        cache_dir (str, optional): The cache dir. If not None, the result will be cached. Defaults to None.

    Returns:
        tuple[ClassifierMixin | RegressorMixin, float]: The trained model and the cpu time of training.
    """
    kwargs = kwargs if kwargs is not None else {}
    model = D_FUNCNAME_FUNC[func_name]
    if model in [XGBClassifier, XGBRegressor]:
        kwargs["random_state"] = seed
        if type(X_train) == pd.DataFrame:
            # category features don't have been encoded
            kwargs.update(
                {
                    "tree_method": "hist",
                    "enable_categorical": True,
                }
            )
    elif model in [LGBMClassifier, LGBMRegressor]:
        kwargs["random_state"] = seed
    elif model in [CatBoostClassifier, CatBoostRegressor]:
        kwargs["verbose"] = False
        kwargs["random_state"] = seed

    model = model(**kwargs)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        model_name = func_name + "_"
        for key, val in kwargs.items():
            model_name += f"{key}={val}_"
        model_name = model_name[:-1] + ".pkl"
        model_path = os.path.join(cache_dir, model_name)
        if os.path.exists(model_path):
            model, time_model = utils.load(model_path)
            logging.info("load from cache: %s", model_path)
        else:
            if seed is not None:
                utils.seed_everything(seed)

            time_start = time.process_time()  # record the cpu time
            _model_fit(model, X_train, y_train, seed=seed)
            time_model = time.process_time() - time_start

            utils.save((model, time_model), model_path)
            logging.info("save to cache: %s", model_path)

    else:
        time_start = time.process_time()  # record the cpu time
        _model_fit(model, X_train, y_train, seed=seed)
        time_model = time.process_time() - time_start

    return model, time_model


def evaluate(
    model: ClassifierMixin | RegressorMixin,
    is_task_reg: bool,
    X: np.ndarray,
    y: np.ndarray,
    suffix: str = "",
) -> dict:
    """Evaluate the model on the test set."""

    if is_task_reg:  # regression
        y_pred = model.predict(X)
        d_metric_val = {
            f"Accuracy{suffix}": None,
            f"F1{suffix}": None,
            f"AUC{suffix}": None,
            f"Precision{suffix}": None,
            f"Recall{suffix}": None,
            f"CrossEntropy{suffix}": None,
            f"ZeroOneLoss{suffix}": None,
            f"HingeLoss{suffix}": None,
            # ↑ classification metrics
            # ↓ regression metrics
            f"MSE{suffix}": metrics.mean_squared_error(y, y_pred),
            f"RMSE{suffix}": metrics.mean_squared_error(y, y_pred, squared=False),
            f"MAE{suffix}": metrics.mean_absolute_error(y, y_pred),
            f"MAPE{suffix}": metrics.mean_absolute_percentage_error(y, y_pred),
            f"R2{suffix}": metrics.r2_score(y, y_pred),
            # 'MSLE': metrics.mean_squared_log_error(y_test, y_pred),
        }
    else:  # classification
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred_proba = y_pred

        d_metric_val = {
            f"Accuracy{suffix}": metrics.accuracy_score(y, y_pred),
            f"F1{suffix}": metrics.f1_score(y, y_pred),
            f"AUC{suffix}": metrics.roc_auc_score(y, y_pred_proba),
            f"Precision{suffix}": metrics.precision_score(y, y_pred),
            f"Recall{suffix}": metrics.recall_score(y, y_pred),
            f"CrossEntropy{suffix}": metrics.log_loss(y, y_pred_proba),
            f"ZeroOneLoss{suffix}": metrics.zero_one_loss(y, y_pred),
            f"HingeLoss{suffix}": metrics.hinge_loss(y, y_pred),
            # ↑ classification metrics
            # ↓ regression metrics
            f"MSE{suffix}": None,
            f"RMSE{suffix}": None,
            f"MAE{suffix}": None,
            f"MAPE{suffix}": None,
            f"R2{suffix}": None,
        }

    return d_metric_val
