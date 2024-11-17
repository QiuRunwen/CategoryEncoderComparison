# -*- coding: utf-8 -*-
"""
Moneyball
It is gathered from baseball-reference.com. The original author 
wanted to predict RS and understand the impact of each variable on it.
The original author wants to know which important factors affect the performance of the baseball team

https://www.openml.org/d/41021

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

    file_path = os.path.join(data_dir, r"OpenML\moneyball", "moneyball.csv")
    df = pd.read_csv(file_path)

    # 1232*15
    # Feature Name	Type	Distinct/Missing Values
    # RS (target)	numeric	374 distinct values 0 missing attributes
    # Team	nominal	39 distinct values 0 missing attributes
    # League	nominal	2 distinct values 0 missing attributes
    # Year	numeric	47 distinct values 0 missing attributes
    # RA	numeric	381 distinct values 0 missing attributes
    # W	numeric	63 distinct values 0 missing attributes
    # OBP	numeric	87 distinct values 0 missing attributes
    # SLG	numeric	162 distinct values 0 missing attributes
    # BA	numeric	75 distinct values 0 missing attributes
    # Playoffs	nominal	2 distinct values 0 missing attributes
    # RankSeason	nominal	8 distinct values 988 missing attributes
    # RankPlayoffs	nominal	5 distinct values 988 missing attributes
    # G	nominal	8 distinct values 0 missing attributes
    # OOBP	numeric	72 distinct values 812 missing attributes
    # OSLG	numeric	112 distinct values 812 missing attributes

    # 2. convert numeric/categorical columns
    cat_cols = ["Team", "League", "Playoffs", "RankSeason", "RankPlayoffs", "G"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "RS"
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
