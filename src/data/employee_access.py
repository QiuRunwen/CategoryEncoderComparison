# -*- coding: utf-8 -*-
"""
The data consists of real historical data collected from 2010 & 2011.
Employees are manually allowed or denied access to resources over time.
The data is used to create an algorithm capable of learning from this historical data
to predict approval/denial for an unseen set of employees.
https://www.kaggle.com/competitions/amazon-employee-access-challenge
@author: QiuRunwen
"""

import os
import pandas as pd
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data

    fp = os.path.join(data_dir, 'kaggle/Amazon_employee_access', 'Amazon_employee_access.csv')
    df = pd.read_csv(fp, sep=',')

    # 32769*10

    # 2. convert numeric/categorical columns

    # ACTION	ACTION is 1 if the resource was approved, 0 if the resource was not
    # RESOURCE	An ID for each resource
    # MGR_ID	The EMPLOYEE ID of the manager of the current EMPLOYEE ID record;
    #           an employee may have only one manager at a time
    # ROLE_ROLLUP_1	Company role grouping category id 1 (e.g. US Engineering)
    # ROLE_ROLLUP_2	Company role grouping category id 2 (e.g. US Retail)
    # ROLE_DEPTNAME	Company role department description (e.g. Retail)
    # ROLE_TITLE	Company role business title description (e.g. Senior Engineering Retail Manager)
    # ROLE_FAMILY_DESC	Company role family extended description (e.g. Retail Manager, Software Engineering)
    # ROLE_FAMILY	Company role family description (e.g. Retail Manager)
    # ROLE_CODE	Company role code; this code is unique to each role (e.g. Manager)

    # ACTION
    # 1    30872
    # 0     1897

    cat_cols = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
                'ROLE_DEPTNAME','ROLE_TITLE','ROLE_FAMILY_DESC','ROLE_FAMILY','ROLE_CODE']
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')

    # 3. simple feature extraction


    # 4. compute class label
    # class        N         N[%]
    # ------------------------------
    # not_recom    4320   (33.333 %)
    # recommend       2   ( 0.015 %)
    # very_recom    328   ( 2.531 %)
    # priority     4266   (32.917 %)
    # spec_prior   4044   (31.204 %)
    y_col = 'ACTION'
    # assert df[y_col].notna().all()

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
    df,y_col = load(verbose=True)
    df.info()
