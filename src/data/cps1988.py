# -*- coding: utf-8 -*-
"""
CPS1988
Cross-section data originating from the March 1988 Current Population Survey
by the US Census Bureau. The data is a sample of men aged 18 to 70
with positive annual income greater than USD 50 in 1992, who are not self-employed nor working without pay.

https://www.openml.org/d/43963

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

    file_path = os.path.join(data_dir, r"OpenML\cps1988", "cps1988.csv")
    df = pd.read_csv(file_path)

    # 28155*7
    # Feature Name	Type	Distinct/Missing Values
    # wage (target)	numeric	5970 distinct values 0 missing attributes
    # education	numeric	19 distinct values 0 missing attributes
    # experience	numeric	67 distinct values 0 missing attributes
    # ethnicity	nominal	2 distinct values 0 missing attributes
    # smsa	nominal	2 distinct values 0 missing attributes
    # region	nominal	4 distinct values 0 missing attributes
    # parttime	nominal	2 distinct values 0 missing attributes

    # wage: Wage (in dollars per week).
    # education: Number of years of education.
    # experience: Number of years of potential work experience.
    # ethnicity: Factor with levels "cauc" and "afam" (African-American).
    # smsa: Factor. Does the individual reside in a Standard Metropolitan Statistical Area (SMSA)?
    # region: Factor with levels "northeast", "midwest", "south", "west".
    # parttime: Factor. Does the individual work part-time?

    # 2. convert numeric/categorical columns
    cat_cols = ["ethnicity", "smsa", "region", "parttime"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "wage"
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
