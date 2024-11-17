
# -*- coding: utf-8 -*-
"""
Dataset comes from kaggle's competition
https://www.kaggle.com/competitions/cat-in-the-dat
300k train, 200k test

https://www.kaggle.com/competitions/cat-in-the-dat-ii
600k train, 400k test

@author: RunwenQiu
"""

import pandas as pd
import os
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir='../../data', is_sec_comp=False, drop_useless=True, num_sample=None, verbose=False):

    # 基本都是预处理好的
    # 1. read/uncompress data
    cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
            'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',
            'day', 'month',
            # 'target'
            ]
    d_col_dtype = {col:'category' for col in cols}
    dfs = []
    filename = 'cat-in-the-dat-ii.zip' if is_sec_comp else 'cat-in-the-dat.zip'
    with ZipFile(os.path.join(data_dir, 'kaggle/cat-int-the-dat', filename)) as zf:
        for name in ['train.csv', 'test.csv']: # 还有'sample_submission.csv'
            with zf.open(name) as f: #
                dfs.append(pd.read_csv(f,index_col='id', dtype=d_col_dtype))


    df_train, df_test = dfs
    y_col = 'target'

    if drop_useless:
        # a. manually remove useless columns

        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df_train)
        df_train = util.drop_useless(df_train, useless_cols_dict,verbose=verbose)

    # 6. sampling by class
    if num_sample is not None:
        df_train = util.sampling_by_class(df_train, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat

    # 原始的df_test是没有label，只能提交到kaggle官网，才能知道分数，不利于实验。因此只返回df_train
    return df_train, y_col


def load1(data_dir='../../data',  *args, **kwargs):
    return load(data_dir, is_sec_comp=False, *args, **kwargs)

def load2(data_dir='../../data', *args, **kwargs):
    return load(data_dir, is_sec_comp=True, *args, **kwargs)

if __name__ == "__main__":
    df_train1, y_col1 = load1(verbose=True)
    df_train1.info()

    df_train2, y_col2 = load2(verbose=True)
    df_train2.info()