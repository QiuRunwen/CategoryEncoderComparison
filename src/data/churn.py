# -*- coding: utf-8 -*-
"""
A dataset relating characteristics of telephony account features and usage and whether or not the customer churned.
https://www.openml.org/d/41283
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

    filepath = os.path.join(data_dir, "OpenML", "churn", "churn.csv")
    with open(filepath) as f:
        df = pd.read_csv(f)

    # 5000*(14+6+1)
    
    # 2. convert numeric/categorical columns
	# Feature Name	Type	Distinct/Missing Values
    # class (target)	nominal	2 distinct values   0 missing attributes
    # state	nominal	51 distinct values  0 missing attributes
    # account_length	numeric	218 distinct values 0 missing attributes
    # area_code	nominal	3 distinct values   0 missing attributes
    # international_plan	nominal	2 distinct values   0 missing attributes
    # voice_mail_plan	nominal	2 distinct values   0 missing attributes
    # number_vmail_messages	numeric	48 distinct values  0 missing attributes
    # total_day_minutes	numeric	1961 distinct values    0 missing attributes
    # total_day_calls	numeric	123 distinct values 0 missing attributes
    # total_day_charge	numeric	1961 distinct values    0 missing attributes
    # total_eve_minutes	numeric	1879 distinct values    0 missing attributes
    # total_eve_calls	numeric	126 distinct values 0 missing attributes
    # total_eve_charge	numeric	1659 distinct values    0 missing attributes
    # total_night_minutes	numeric	1853 distinct values    0 missing attributes
    # total_night_calls	numeric	131 distinct values 0 missing attributes
    # total_night_charge	numeric	1028 distinct values    0 missing attributes
    # total_intl_minutes	numeric	170 distinct values 0 missing attributes
    # total_intl_calls	numeric	21 distinct values  0 missing attributes
    # total_intl_charge	numeric	170 distinct values 0 missing attributes
    # number_customer_service_calls	nominal	10 distinct values  0 missing attributes


    cat_cols = ["state", "area_code", "international_plan", "voice_mail_plan"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    # df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit="s")

    # 4. compute class label

    y_col = "class"
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
