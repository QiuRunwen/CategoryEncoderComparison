# -*- coding: utf-8 -*-
"""
Datasets from ACM KDD Cup 2009
https://www.openml.org/d/1114
The KDD Cup 2009 offers the opportunity to work on large marketing databases from the French Telecom company Orange
to predict the propensity of customers to switch provider (churn), buy new products or services (appetency), 
or buy upgrades or add-ons proposed to them to make the sale more profitable (up-selling)

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

    fp = os.path.join(data_dir, 'OpenML/KDDCup09_upselling', 'KDDCup09_upselling.csv')
    df = pd.read_csv(fp, sep=',')

    # 50,000 *231
    # 2. convert numeric/categorical columns

    # The first predictive 190 variables are numerical and the last 40 predictive variables are categorical.
    # The last target variable is binary {-1,1}.

    cat_cols = [f'Var{i}' for i in range(191, 231)]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')

    # 3. simple feature extraction


    # 4. compute class label
    # UPSELLING
    # 0    46318
    # 1     3682
    y_col = 'UPSELLING'
    df['UPSELLING'] = df['UPSELLING'].map({1:1,-1:0})
    # assert df[y_col].notna().all()

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        empty_cols = ['Var8', 'Var15', 'Var20', 'Var31', 'Var32', 'Var39', 'Var42',
                      'Var48', 'Var52', 'Var55', 'Var79', 'Var141', 'Var167', 'Var169',
                      'Var175', 'Var185', 'Var209', 'Var230']
        single_value_cols = ['Var118', 'Var191', 'Var213', 'Var215', 'Var224']
        too_many_missing_cols = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var9', 'Var10', 'Var11', 
                                 'Var12', 'Var14', 'Var16', 'Var17', 'Var18', 'Var19', 'Var23', 
                                 'Var26', 'Var27', 'Var29', 'Var30', 'Var33', 'Var34', 'Var36', 
                                 'Var37', 'Var40', 'Var41', 'Var43', 'Var45', 'Var46', 'Var47', 
                                 'Var49', 'Var50', 'Var51', 'Var53', 'Var54', 'Var56', 'Var58', 
                                 'Var59', 'Var60', 'Var61', 'Var62', 'Var63', 'Var64', 'Var66', 
                                 'Var67', 'Var68', 'Var69', 'Var70', 'Var71', 'Var75', 'Var77', 
                                 'Var80', 'Var82', 'Var84', 'Var86', 'Var87', 'Var88', 'Var89', 
                                 'Var90', 'Var91', 'Var92', 'Var93', 'Var95', 'Var96', 'Var97', 
                                 'Var98', 'Var99', 'Var100', 'Var101', 'Var102', 'Var103', 
                                 'Var104', 'Var105', 'Var106', 'Var107', 'Var108', 'Var110', 
                                 'Var111', 'Var114', 'Var115', 'Var116', 'Var117', 'Var120', 
                                 'Var121', 'Var122', 'Var124', 'Var127', 'Var128', 'Var129', 
                                 'Var130', 'Var131', 'Var135', 'Var136', 'Var137', 'Var138', 'Var139', 
                                 'Var142', 'Var145', 'Var146', 'Var147', 'Var148', 'Var150', 'Var151', 
                                 'Var152', 'Var154', 'Var155', 'Var156', 'Var157', 'Var158', 'Var159', 
                                 'Var161', 'Var162', 'Var164', 'Var165', 'Var166', 'Var168', 'Var170', 
                                 'Var171', 'Var172', 'Var174', 'Var176', 'Var177', 'Var178', 'Var179', 
                                 'Var180', 'Var182', 'Var183', 'Var184', 'Var186', 'Var187', 'Var188', 
                                 'Var189', 'Var190', 'Var194', 'Var200', 'Var201', 'Var214', 'Var225', 'Var229']
        too_large_cat_cols = ['Var195', 'Var196', 'Var203', 'Var208', 'Var210']
        df.drop(columns=empty_cols+single_value_cols+too_many_missing_cols+too_large_cat_cols, inplace=True)
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