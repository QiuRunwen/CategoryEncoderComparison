# -*- coding: utf-8 -*-
"""
暂时没用, 有raw和处理后的数据
raw数据中, y_col的比例 过于悬殊
loan_status
0    840151
1     47228

0    200160
1     43831

如果是 2007-2015年的所有数据，大概 25 万条
"""


import os
from zipfile import ZipFile
import pandas as pd

if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用




def load(data_dir='../../data', drop_useless=True, num_sample=None, verbose=False):

    # 1. read/uncompress data
    # with ZipFile(os.path.join(data_dir, 'TianChi/lending-club-loan-data/loan_2007-2015.zip')) as zf:
    #     with zf.open('loan_2007-2015.csv') as f:
    #         df = pd.read_csv(f, index_col=0)

    with ZipFile(os.path.join(data_dir, 'TianChi/lending-club-loan-data/loan.zip')) as zf:
        with zf.open('loan.csv') as f:
            df = pd.read_csv(f, index_col='id')


    # 2. convert numeric/categorical columns
    num_vars = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'emp_length',
                'dti', 'delinq_2yrs', 'inq_last_6mths','pub_rec',
                'total_acc', 'open_acc']

    cat_vars = ['zip_code','verification_status','term', 'grade','home_ownership',
                'purpose','initial_list_status', 'loan_status']

    # for col in num_vars:
    #     if not pd.api.types.is_numeric_dtype(df[col]):
    #         df[col] = df[col].to_numeric()

    for col in cat_vars:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')

    # 3. simple feature extraction

    # 4. compute class label
    #

    # Name: loan_status, dtype: int64

    # | loan_status                                           | 描述                   | 划分           |
    # | ----------------------------------------------------- | ---------------------- | -------------- |
    # | current                                               | 贷中，还在正常还款当中 | Cur，贷中      |
    # | Issued                                                | 发布                   | Cur，贷中      |
    # | Fully Paid                                            | 全额还款，已结清       | Good，完成还款 |
    # | Does not meet  the credit policy. Status: Fully Paid  | 全额还款（不符合信用） | Good，完成还款 |
    # | In Grace  Period                                      | 宽限期（逾期15天之内） | Late，逾期     |
    # | Late (16-30  days)                                    | 逾期16-30天            | Late，逾期     |
    # | Late (31-120  days)                                   | 逾期31-120天           | Late，逾期     |
    # | Default                                               | 违约                   | Bad，不良/坏账 |
    # | Charged Off                                           | 坏账，注销             | Bad，不良/坏账 |
    # | Does not meet  the credit policy. Status: Charged Off | 坏账（不符合信用）     | Bad，不良/坏账 |

    y_col = 'loan_status'
    df[y_col] = df[y_col].transform(lambda x: 1 if x in
                                    ['Default','Does not meet the credit policy. Status:Charged Off'
                                     ,'Charged Off']
                                    else 0)
    # df[y_col] = df[y_col].astype(int)

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