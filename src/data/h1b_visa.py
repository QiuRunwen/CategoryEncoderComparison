# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:28:00 2022

@author: YingFu, Wenbin Zhu
"""

import pandas as pd
import os
from zipfile import ZipFile
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 
    
def load(data_dir='../../data', drop_useless=True, num_sample=600000, verbose=False): 
    
    # 1. read/uncompress data
    with ZipFile(os.path.join(data_dir, 'kaggle/H1BVisa/h1b_kaggle.csv.zip')) as zf:
        with zf.open('h1b_kaggle.csv') as f:
            df = pd.read_csv(f, index_col=0)
            
    # print(df.info())
    
    #  #   Column              Dtype   
    # ---  ------              -----   
    #  0   Unnamed: 0          int64        index
    #  1   CASE_STATUS         category     7
    #  2   EMPLOYER_NAME       category     236013
    #  3   SOC_NAME            category     2132    Standard Occupitional Classification
    #  4   JOB_TITLE           category     287549
    #  5   FULL_TIME_POSITION  category     2 (2576111,426332)
    #  6   PREVAILING_WAGE     float64 
    #  7   YEAR                float64      6 (358767, ..., 647803)
    #  8   WORKSITE            category     18622
    #  9   lon                 float64 
    #  10  lat                 float64 
    
    # 2. convert numeric/categorical columns
    cat_cols = ['CASE_STATUS','EMPLOYER_NAME','SOC_NAME',
                'JOB_TITLE','FULL_TIME_POSITION','WORKSITE', 'YEAR'] # YEAR 2011 - 2016 当作cat_var处理
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    # print(df.info())
    # print(f"{df['EMPLOYER_NAME'].value_counts().size=}")


    # 3. simple feature extraction

    # 4. compute class label
    #
    # print(df['CASE_STATUS'].value_counts(dropna=False))
    # CERTIFIED                                             2615623
    # CERTIFIED-WITHDRAWN                                    202659
    # DENIED                                                  94346
    # WITHDRAWN                                               89799
    # PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED         15
    # NaN                                                        13
    # REJECTED                                                    2
    # INVALIDATED                                                 1
    # 2,615,623 vs 386,822
    df = df[df['CASE_STATUS'].notna()]  # delete records with unknown target_value
    y_col = 'CERTIFIED'
    df[y_col] = df['CASE_STATUS'].transform(lambda x: 1 if x=='CERTIFIED' else 0)
    df.drop(columns='CASE_STATUS', inplace=True) # delete CASE_STATUS

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # 未标准化的文字需要特殊处理，不在本研究讨论范围内
        df.drop(columns=['EMPLOYER_NAME','JOB_TITLE','WORKSITE'], inplace=True) 

        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)
    
    # 6. sampling by class
    if num_sample is not None:
        
        df = util.sampling_by_class(df, y_col, num_sample=num_sample)
        
        # remove categorical cols with too few samples_per_cat
        #
        # 这是 small sample 的数据统计，因为数据只有 3W 条， 高基量不宜选得过大，因此选择的高基变量是 SOC_NAME
        # SOC_NAME 表示 Standard Occupational Classification，美国标准职业分类系统
        # 因为只有三万条数据，删掉 samples_per_cat 过少的列（EMPLOYER_NAME, JOB_TITLE）
        #                 	cardinality	samples_per_cat	importance_rank	importance_rank_cat
        # EMPLOYER_NAME	      13157	     2.280155051	1	               1
        # SOC_NAME	           763	    39.31847969	    5	               4
        # JOB_TITLE	           9594	     3.126954346	2	               2
        # FULL_TIME_POSITION	  2	       15000	    9	               5
        # PREVAILING_WAGE			                    4	
        # YEAR			                                8	
        # WORKSITE	           3034	   9.887936717	    3	               3
        # lon			                                7	
        # lat			                                6	
        # TODO: EMPLOYER_NAME 做成衍生变量，比如这个公司是否有先例是成功的，成功率是多少
        # df.drop(columns='EMPLOYER_NAME', inplace=True) 
        # df.drop(columns='JOB_TITLE', inplace=True) 
        # print(df['SOC_NAME'].value_counts(dropna=False)) # will show count of zero for some variables
    
    return df,y_col


if __name__ == "__main__":
    import time
    start_time = time.time()
    df,y_col = load(verbose=True)
    time_cost = time.time() - start_time # 没有用cpu time，因为IO也是挺花时间的
    print(f'`load` spend {time_cost}s')
    df.info()