# -*- coding: utf-8 -*-
"""
Las Vegas Strip

This dataset includes quantitative and categorical features 
from online reviews from 21 hotels located in Las Vegas Strip, 
extracted from TripAdvisor (http://www.tripadvisor.com).

https://archive.ics.uci.edu/dataset/397/las+vegas+strip

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

    file_path = os.path.join(
        data_dir, r"UCI\las+vegas+strip", "LasVegasTripAdvisorReviews-Dataset.csv"
    )
    df = pd.read_csv(file_path, sep=";")

    # 504*20

    # 2. convert numeric/categorical columns
    cat_cols = [
        col
        for col in df.columns
        if col
        not in [
            "Nr. reviews",
            "Nr. hotel reviews",
            "Helpful votes",
            "Score",
            "Member years",
        ]
    ]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # Score
    # 5    227
    # 4    164
    # 3     72
    # 2     30
    # 1     11

    y_col = "Score"
    df[y_col] = df[y_col].map(lambda x: 1 if x == 5 else 0)
    assert df[y_col].notna().all()

    # after convert to binary class
    # Score
    # 0    277
    # 1    227

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
