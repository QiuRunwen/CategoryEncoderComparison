# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:12:54 2022

It includes annual salary information for 2016 for Montgomery County, Maryland employees,
downloaded from https://catalog.data.gov/dataset/employee-salaries-2016

@author: YingFu, RunwenQiu
"""

import pandas as pd
import os

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_name = os.path.join(data_dir, "data.gov/employee_salaries/rows.csv")
    df = pd.read_csv(file_name)
    #  #   Column                   Non-Null Count  Dtype
    # ---  ------                   --------------  -----
    #  0   Full Name                9228 non-null   category
    #  1   Gender                   9211 non-null   category
    #  2   Current Annual Salary    9228 non-null   float64
    #  3   2016 Gross Pay Received  9128 non-null   float64
    #  4   2016 Overtime Pay        6311 non-null   float64
    #  5   Department               9228 non-null   category
    #  6   Department Name          9228 non-null   category
    #  7   Division                 9228 non-null   category
    #  8   Assignment Category      9228 non-null   category
    #  9   Employee Position Title  9228 non-null   category
    #  10  Underfilled Job Title    1093 non-null   category
    #  11  Date First Hired         9228 non-null   datetime64[ns]

    # 2. convert numeric/categorical columns
    cat_cols = [
        "Full Name",
        "Gender",
        "Department",
        "Department Name",
        "Division",
        "Assignment Category",
        "Employee Position Title",
        "Underfilled Job Title",
    ]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    #
    # Use the date of latest hired employee as baseline, compute the number of
    # days each employeed has worked before the baseline.
    # 用最后聘请的人的日期作为基准，计算每个人在基准日之前已经工作的天数
    time_cols = ["Date First Hired"]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    most_recent_date = df["Date First Hired"].max()
    # print(f"{most_recent_date = }")
    df["working days"] = (most_recent_date - df["Date First Hired"]).dt.days
    df.drop(columns="Date First Hired", inplace=True)

    # print(df.info())

    # 4. compute class label
    y_col = "Current Annual Salary"
    # df[y_col] = df['Current Annual Salary'].apply(lambda x: 1 if x>=100000 else 0)
    # df.drop(columns='Current Annual Salary', inplace=True)

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # Department 是 Department Name的缩写，两者一一对应，删掉其中一列
        # print(f"{util.is_one_to_one(df, 'Department', 'Department Name')=}")  # Output one-to-one; or None
        df.drop(columns="Department", inplace=True)

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
