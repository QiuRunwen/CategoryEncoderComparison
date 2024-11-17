# -*- coding: utf-8 -*-
"""
Cholesterol

The earliest one is from [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease),
which is a classification task. The target is to predict whether a patient has heart disease.
The target variable is `num` (0: no heart disease, 1-4: heart disease).

In OpenML, the dataset is called [cholesterol](https://www.openml.org/d/204).
The target variable is `chol` (serum cholestoral in mg/dl), which is a regression task.

@author: QiuRunwen
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data

    file_path = os.path.join(data_dir, r"OpenML\cholesterol", "cholesterol.csv")
    df = pd.read_csv(file_path)

    # 303*14
    # Feature Name	Type	Distinct/Missing Values
    # chol (target)	numeric	152 distinct values 0 missing attributes
    # age	numeric	41 distinct values 0 missing attributes
    # sex	nominal	2 distinct values 0 missing attributes
    # cp	nominal	4 distinct values 0 missing attributes
    # trestbps	numeric	50 distinct values 0 missing attributes
    # fbs	nominal	2 distinct values 0 missing attributes
    # restecg	nominal	3 distinct values 0 missing attributes
    # thalach	numeric	91 distinct values 0 missing attributes
    # exang	nominal	2 distinct values 0 missing attributes
    # oldpeak	numeric	40 distinct values 0 missing attributes
    # slope	nominal	3 distinct values 0 missing attributes
    # ca	numeric	4 distinct values 4 missing attributes
    # thal	nominal	3 distinct values 2 missing attributes
    # num	numeric	5 distinct values 0 missing attributes

    # 2. convert numeric/categorical columns
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "chol"
    assert df[y_col].notna().all()

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
