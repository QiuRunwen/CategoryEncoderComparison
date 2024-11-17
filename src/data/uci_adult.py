# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:31:58 2021

Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
https://archive.ics.uci.edu/dataset/2/adult

@author: iwenc, QiuRunwen
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_name = os.path.join(data_dir, "UCI/Adult/adult.data")
    headers = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "predclass",
    ]
    training_set = pd.read_csv(
        file_name,
        header=None,
        names=headers,
        sep=r",\s",
        na_values=["?"],
        on_bad_lines="error",
        engine="python",
    )

    file_name = os.path.join(data_dir, "UCI/Adult/adult.test")
    test_set = pd.read_csv(
        file_name,
        header=None,
        names=headers,
        sep=r",\s",
        na_values=["?"],
        on_bad_lines="error",
        engine="python",
        skiprows=1,
    )

    df = pd.concat([training_set, test_set])

    # VariableName	Type	Demographic Description	Units	MissingValues
    # age	Integer	Age	N/A		no
    # workclass	Categorical	Income	Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, ....		yes
    # fnlwgt	Integer				no
    # education	Categorical	Education Level	Bachelors, Some-college, 11th, HS-grad, Prof-school, ...		no
    # education-num	Integer	Education Level			no
    # marital-status	Categorical	Other	Married-civ-spouse, Divorced, Never-married, ...		no
    # occupation	Categorical	Other	Tech-support, Craft-repair, Other-service, ...		yes
    # relationship	Categorical	Other	Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.		no
    # race	CategoricalRace	White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.		no
    # sex	Binary	Sex	Female, Male.		no
    # capital-gain	Integer				no
    # capital-loss	Integer				no
    # hours-per-week	Integer				no
    # native-country	Categorical	Other	United-States, Cambodia, England, Puerto-Rico, ...		yes
    # income	Target	Binary	Income	>50K, <=50K.		no

    # 2. convert numeric/categorical columns
    num_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].to_numeric()
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # predclass
    # <=50K     24720
    # <=50K.    12435
    # >50K       7841
    # >50K.      3846
    # Name: count, dtype: int64

    y_col = "predclass"
    df[y_col] = df[y_col].map(lambda y: 1 if y == ">50K" or y == ">50K." else 0)

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns

        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)

    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat

    return df, y_col


if __name__ == "__main__":
    df, y_col = load(verbose=True)
    df.info()
