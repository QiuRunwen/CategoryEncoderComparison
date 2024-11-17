# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:58:09 2021

@author: iwenc
"""

import os
import pandas as pd
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 


def load(data_dir='../../data', drop_useless=True, num_sample=None, verbose=False):
    
    # 1. read/uncompress data
    with ZipFile(os.path.join(data_dir, 'TianChi/Auto-loan-default-risk/data.zip')) as zipfile:
        with zipfile.open('data.csv') as f:
            df = pd.read_csv(f)

    
    # 2. convert numeric/categorical columns
    cat_cols = ['customer_id','branch_id','supplier_id',
                'manufacturer_id','area_id','employee_code_id', 
               'mobileno_flag', 'idcard_flag','Driving_flag','passport_flag',
               'employment_type']
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')


    # 3. simple feature extraction

        
    # 4. compute class label
    y_col = 'loan_default'
    
    # print(f"{df[y_col].value_counts()=}")
    # # 0： 164289
    # # 1： 35428


    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # print(f"year_of_birht+age counts: {(df['year_of_birth']+df['age']).value_counts().size}")
        df.drop(columns='year_of_birth', inplace=True) # always = 2019 - df['age'], redundant
        
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)    

    
    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat

    # TODO: fix the bug due to combination of SimpleImputer and TargetEncoding
    df['branch_id'] = df['branch_id'].astype('str').astype('category')

    return df, y_col

if __name__ == "__main__":
    df,y_col = load(verbose=True)
    df.info()