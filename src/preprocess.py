"""
预处理数据
"""

# import sys

import logging
import time
from types import MappingProxyType

import numpy as np
import pandas as pd
import category_encoders
import dirty_cat

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mean_encoder import MeanEncoder
from delete_encoder import DeleteEncoder
from datasets import Dataset, calc_col_card, get_num_cat_cols
import utils

UNSUPERVISED_ENCODERS = (
    category_encoders.OneHotEncoder,
    category_encoders.OrdinalEncoder,
    category_encoders.CountEncoder,
    category_encoders.BaseNEncoder,
    category_encoders.BackwardDifferenceEncoder,
    category_encoders.HelmertEncoder,
    category_encoders.SumEncoder,
    # category_encoders.GrayEncoder, # 这个不适用，分类型变量必须要是有顺序的
    # category_encoders.PolynomialEncoder, # 这个不适用，分类型变量必须要是有顺序的，不然会报错
    # category_encoders.HashingEncoder, # 只适用于文本类型
    dirty_cat.MinHashEncoder,
    dirty_cat.SimilarityEncoder,
    DeleteEncoder,
)

SUPERVISED_ENCODERS = (
    category_encoders.TargetEncoder,
    category_encoders.LeaveOneOutEncoder,  # 文档中最优值是0.05~0.6
    category_encoders.CatBoostEncoder,
    category_encoders.GLMMEncoder,  # O(p^3 + Np^3) p是特征数，N是样本数
    category_encoders.WOEEncoder,  # 只适用于二分类，不适用于回归
    category_encoders.JamesSteinEncoder,
    category_encoders.MEstimateEncoder,
    category_encoders.QuantileEncoder,  # 这个不适用于二分类，否则会只编码成0,1,分位数值
    # category_encoders.SummaryEncoders,  # 多次的QuantileEncoder
    MeanEncoder,
)

DICT_NAME_ENCODER = MappingProxyType(
    {
        encoder.__name__: encoder
        for encoder in UNSUPERVISED_ENCODERS + SUPERVISED_ENCODERS
    }
)


def get_default_name_kwargs() -> dict:
    """Get the default name and kwargs of encoders."""
    res = {}
    for func_name in DICT_NAME_ENCODER:
        res[func_name] = {
            "func_name": func_name,
            "kwargs": {},
        }
    return res


def get_available_encoder_name(target_type: str = None) -> list[str]:
    """Get the names of available encoders."""
    names = [
        name
        for name in DICT_NAME_ENCODER.keys()
        if name
        not in ["GrayEncoder", "PolynomialEncoder", "HashingEncoder", "SummaryEncoder"]
    ]
    names.append("Delete")
    if target_type is None:
        return names

    if target_type == "binomial":
        return [
            name for name in names if name not in ["QuantileEncoder", "SummaryEncoder"]
        ]

    if target_type == "continuous":
        return [name for name in names if name not in ["WOEEncoder"]]

    raise ValueError(f"target_type: {target_type} is not supported.")


def _format_dataset(dataset: Dataset):
    df = dataset.df
    y_col = dataset.y_col
    X = df.drop(columns=[y_col])
    y = df[y_col].copy()
    if y.unique().shape[0] == 2:
        y = y.astype("int")  # for encoding
    num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols = get_num_cat_cols(
        df, y_col
    )
    # TODO category_encoders的函数默认只对object的array转换(有指明category的df也行)，
    # 而在imputer返回的是原数据类型的array，如果原先dtype全为数值，则category_encoders失效。因此数值型的category最好转换下
    # X[cat_is_num_cols] = X[cat_is_num_cols].astype(str).astype("category")
    # X[cat_miss_lable_cols] = X[cat_miss_lable_cols].astype(str).astype("category")
    X[cat_cols] = X[cat_cols].astype(str).astype("category")

    for col in num_cols:
        col_max = X[col].max()
        if col_max == np.inf:
            logging.warning("Data set: %s col: %s np.inf -> max+1", df.shape, col)
            new_col = X[col].replace([np.inf], np.nan)
            col_max = new_col.max()
            X[col].replace([np.inf], col_max + 1, inplace=True)

        col_min = X[col].min()
        if col_min == -np.inf:
            logging.warning("Data set: %s col: %s -np.inf -> min-1", df.shape, col)
            new_col = X[col].replace([-np.inf], np.nan)
            col_min = new_col.min()
            X[col].replace([-np.inf], col_min - 1, inplace=True)
    return X, y, num_cols, cat_cols


def split_impute_encode_scale(
    dataset: Dataset,
    test_size: float,
    encoder_func_name: str,
    encoder_kwargs: dict = None,
    seed: int = None,
    GLMM_sample=10000000,  # not used for now. so set a large number
    GLMM_max_effect_card=1600000,  # not used for now. so set a large number
    delete_highest: bool = False,
    # encoder_cache_dir:str=None, # TODO
) -> tuple[
    np.ndarray | pd.DataFrame,
    np.ndarray | pd.DataFrame,
    np.ndarray,
    np.ndarray,
    float,
]:
    time_start = time.process_time()
    df_X, ser_y, num_cols, cat_cols = _format_dataset(dataset)
    X_train_df, X_test_df, y_train_ser, y_test_ser = train_test_split(
        df_X, ser_y, test_size=test_size, random_state=seed
    )
    if delete_highest:
        tmp_cat_cols = X_train_df.select_dtypes(include=["category"]).columns
        if len(tmp_cat_cols) == 0:
            raise ValueError("No categorical column.")

        col = X_train_df[tmp_cat_cols].nunique().sort_values(ascending=False).index[0]
        cat_cols.remove(col)
        X_train_df = X_train_df[num_cols + cat_cols]
        X_test_df = X_test_df[num_cols + cat_cols]

    if encoder_func_name in ("Auto", "Delete", "DeleteEncoder"):
        encoder = None
    else:
        encoder = DICT_NAME_ENCODER[encoder_func_name](**encoder_kwargs)
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if encoder is not None:
        cat_steps.append(("encoder", encoder))
        cat_steps.append(("scaler", StandardScaler()))  # 很多涉及到距离算法，所以标准化会好些

    cat_pipeline = Pipeline(steps=cat_steps)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Warning 注意trans这步后会改变col的顺序
    trans = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    # if encoder_func_name in ["Auto", "Delete"]:
    #     trans = trans.set_output(
    #         transform="pandas"
    #     )  # return pandas.DataFrame instead of numpy.ndarray
    #     # trans will keep the column names, but the order is not guaranteed
    #     # the order is the same as the order of the columns
    #     # in the ColumnTransformer(transformers=[...]), num_cols and cat_cols

    if seed is not None:
        utils.seed_everything(seed)  # some encoder need random seed

    if isinstance(encoder, category_encoders.GLMMEncoder):
        # GLMMEncoder is too slow for large dataset that has large row and high cardinality.
        effect_card = calc_col_card(X_train_df)
        if effect_card.fillna(1).sum() > GLMM_max_effect_card:
            GLMM_sample = GLMM_max_effect_card
        sample_rate = GLMM_sample / X_train_df.shape[0]
        #     sample_rate = 0.25
        # else:
        #     sample_rate = 1
        if sample_rate < 1:
            X_train_df, y_train_ser = utils.random_sampling(
                X=X_train_df, y=y_train_ser, sample_rate=sample_rate, seed=seed
            )
            trans.fit(X_train_df, y_train_ser)
            X_train = trans.transform(X_train_df)
        else:
            trans.fit(X_train_df, y_train_ser)
            X_train = trans.transform(X_train_df)

    elif isinstance(
        encoder,
        (
            category_encoders.LeaveOneOutEncoder,
            category_encoders.CatBoostEncoder,
        ),
    ):
        # LeaveOneOutEncoder, CatBoostEncoder在training set必须fit_transform. 不然会将其视为test set处理。
        X_train = trans.fit_transform(X_train_df, y_train_ser)
    else:
        trans.fit(X_train_df, y_train_ser)
        X_train = trans.transform(X_train_df)
    X_test = trans.transform(X_test_df)

    # keep the column names and order if dimention is not changed
    # if encoder_func_name in ["Auto", "Delete"]:
    #     X_train.columns = X_train.columns.str.replace(r"num__|cat__", "", regex=True)
    #     X_train = X_train[X_train_df.columns]
    #     X_test.columns = X_test.columns.str.replace(r"num__|cat__", "", regex=True)
    #     X_test = X_test[X_test_df.columns]

    y_train = y_train_ser.to_numpy()
    y_test = y_test_ser.to_numpy()

    if dataset.is_task_reg:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.reshape(-1, 1))
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    time_process = time.process_time() - time_start

    # for Auto
    # if encoder is None:
    #     # 因为ColumnTransformer改变了col的顺序，所以这里的columns只能是对应的num_cols和cat_cols
    #     X_train = pd.DataFrame(X_train, columns=num_cols + cat_cols)
    #     X_test = pd.DataFrame(X_test, columns=num_cols + cat_cols)
    #     X_train[num_cols] = X_train[num_cols].astype(float)
    #     X_train[cat_cols] = X_train[cat_cols].astype(str).astype("category")
    #     X_test[num_cols] = X_test[num_cols].astype(float)
    #     X_test[cat_cols] = X_test[cat_cols].astype(str).astype("category")
    # cat_cols_bool = [(col in cat_cols) for col in cols]
    # return X_train, y_train, X_test, y_test, time_process, cat_cols, cat_cols_bool

    if encoder_func_name == "Auto":
        # keep DataFrame
        X_train = pd.DataFrame(X_train, columns=num_cols + cat_cols)
        X_test = pd.DataFrame(X_test, columns=num_cols + cat_cols)
        X_train[num_cols] = X_train[num_cols].astype(float)
        X_train[cat_cols] = X_train[cat_cols].astype(str).astype("category")
        X_test[num_cols] = X_test[num_cols].astype(float)
        X_test[cat_cols] = X_test[cat_cols].astype(str).astype("category")
    elif encoder_func_name in ("Delete", "DeleteEncoder"):
        X_train = pd.DataFrame(X_train, columns=num_cols + cat_cols)
        X_test = pd.DataFrame(X_test, columns=num_cols + cat_cols)
        if len(num_cols) == 0:  # no numeric column
            X_train = None
            X_test = None
        else:
            X_train = X_train[num_cols].to_numpy()
            X_test = X_test[num_cols].to_numpy()
    else:
        # keep numpy
        X_train = X_train
        X_test = X_test

    return X_train, X_test, y_train, y_test, time_process
