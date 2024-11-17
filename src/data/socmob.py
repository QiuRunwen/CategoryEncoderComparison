# -*- coding: utf-8 -*-
"""
socmob
This dataset described social mobility, i.e. how the sons' occupations are related to their fathers' jobs.
An instance represent the number of sons that have a certain job A given the father has the job B
(additionally conditioned on race and family structure).
The dataset was originally collected for the survey of "Occupational Change in a Generation II". The version 
we use here is the one from OpenML, which is a preprocessed version of the original dataset.

https://www.openml.org/d/44987

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

    file_path = os.path.join(data_dir, r"OpenML\socmob", "socmob.csv")
    df = pd.read_csv(file_path)

    # 1156*2
    # Feature Name	Type	Distinct/Missing Values
    # counts_for_sons_current_occupation (target)	numeric	361 distinct values 0 missing attributes
    # fathers_occupation	nominal	17 distinct values 0 missing attributes
    # sons_occupation	nominal	17 distinct values 0 missing attributes
    # family_structure	nominal	2 distinct values 0 missing attributes
    # race	nominal	2 distinct values 0 missing attributes
    # counts_for_sons_first_occupation	numeric	358 distinct values 0 missing attributes

    # 2. convert numeric/categorical columns
    cat_cols = ["fathers_occupation", "sons_occupation", "family_structure", "race"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "counts_for_sons_current_occupation"
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
