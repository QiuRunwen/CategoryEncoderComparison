# -*- coding: utf-8 -*-
"""
chscase_foot

https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=lte_1&id=703

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

    file_path = os.path.join(data_dir, r"OpenML\chscase_foot", "chscase_foot.csv")
    df = pd.read_csv(file_path)

    # 526*6
    # Feature Name	Type	Distinct/Missing Values
    # col_6 (target)	numeric	3 distinct values 0 missing attributes
    # col_1	nominal	297 distinct values 0 missing attributes
    # col_2	nominal	3 distinct values 0 missing attributes
    # col_3	numeric	5 distinct values 0 missing attributes
    # col_4	numeric	2 distinct values 0 missing attributes
    # col_5	numeric	17 distinct values 0 missing attributes

    # 2. convert numeric/categorical columns
    cat_cols = ["col_1", "col_2"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "col_6"
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
