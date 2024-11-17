# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:53:35 2021

@author: Xiaoting Wu, Runwen Qiu, Wenbin Zhu
"""

import pandas as pd
import numpy as np
import os
from zipfile import ZipFile

if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 


def load(data_dir="../../data", drop_useless=True, num_sample=200000, verbose=False):
    # 1. read/uncompress data
    filename = os.path.join(data_dir, 'LACity/Crime/Crime_Data_from_2010_to_Present.zip')
    with ZipFile(filename) as zf:
        with zf.open('Crime_Data_from_2010_to_Present.csv') as file:
            df = pd.read_csv(file, dtype={'TIME OCC':'str'}, low_memory = False)

    # Remove the trailing space in column name 'AREA '    
    df.rename(columns=lambda x:x.strip(),inplace=True)
    
    #  #   Column          Non-Null Count    Dtype  
    # ---  ------          --------------    -----  
    #  0   DR_NO           1074524 non-null  int64          int,id 
    #  1   Date Rptd       1074524 non-null  datetime64[ns]
    #  2   DATE OCC        1074524 non-null  datetime64[ns] 
    #  3   TIME OCC        1074524 non-null  int64          'hhmm'
    #  4   AREA            1074524 non-null  int64          cat(21) 
    #  5   AREA NAME       1074524 non-null  object         cat(21)
    #  6   Rpt Dist No     1074524 non-null  int64          cat(21)
    #  7   Part 1-2        1074524 non-null  int64          int
    #  8   Crm Cd          1074524 non-null  int64          cat(136)
    #  9   Crm Cd Desc     1074524 non-null  object         cat(136)
    #  10  Mocodes         956809 non-null   object         space separated list of codes
    #  11  Vict Age        1074524 non-null  int64          int
    #  12  Vict Sex        976996 non-null   object         cat(5) 
    #  13  Vict Descent    976976 non-null   object         cat(20)
    #  14  Premis Cd       1074486 non-null  float64        cat(223)
    #  15  Premis Desc     1074486 non-null  object         cat(222)
    #  16  Weapon Used Cd  355327 non-null   float64        cat(79)
    #  17  Weapon Desc     355326 non-null   object         cat(78)
    #  18  Status          1074522 non-null  object         cat(8)
    #  19  Status Desc     1074524 non-null  object         cat(6)
    #  20  Crm Cd 1        1074520 non-null  float64        cat(141)
    #  21  Crm Cd 2        66341 non-null    float64        cat(127)
    #  22  Crm Cd 3        1161 non-null     float64        cat(46)
    #  23  Crm Cd 4        41 non-null       float64        cat(8)
    #  24  LOCATION        1074524 non-null  object         text, addr
    #  25  Cross Street    176913 non-null   object         text, addr
    #  26  LAT             1074524 non-null  float64        float64
    #  27  LON             1074524 non-null  object         num

    # LAT LON, (0, 0) is NAN
    df['LAT'].replace(0, np.nan,inplace=True)
    df['LON'].replace(0, np.nan,inplace=True)
    
    # convert 'hhmm' to minutes since midnight
    df['TIME OCC'] = df['TIME OCC'].transform(lambda x: int(x[:2])*60+int(x[-2:]))

    # last row, LON = '-11{'  seems to be an incomplete record, drop it
    df.drop(index=df.index[-1], inplace=True)
    df['LON'] = df['LON'].astype('float64')    
    cols_date = ['Date Rptd', 'DATE OCC']
    for col in cols_date:
        df[col] = pd.to_datetime(df[col],format='%m/%d/%Y %H:%M:%S AM')

        
    # 2. convert numeric/categorical columns
    cols_cat = ['AREA','AREA NAME','Rpt Dist No','Crm Cd','Crm Cd Desc',
        'Vict Sex', 'Vict Descent','Premis Cd','Premis Desc','Weapon Used Cd',
        'Weapon Desc','Status','Status Desc','Crm Cd 1','Crm Cd 2','Crm Cd 3',
        'Crm Cd 4']
    for col in cols_cat:
        df[col] = df[col].astype('category')
        
    cols_text = ['LOCATION','Cross Street']


    # 3. simple feature extraction
    # feature extraction
    df['report_lag'] = df['Date Rptd'] - df['DATE OCC']
    df['report_lag'] = df['report_lag'].dt.days
    # 留下 'DATE OCC' 的年月日，去掉 时分秒（无效的）
    df['Year OCC'] = df['DATE OCC'].dt.year
    df['Month OCC'] = df['DATE OCC'].dt.month
    df['Day OCC'] = df['DATE OCC'].dt.day
    df['Day Of Week OCC'] = df['DATE OCC'].dt.dayofweek
    df['Week Of Year OCC'] = df['DATE OCC'].dt.isocalendar().week
    # drop two date column
    df.drop(columns=['DATE OCC','Date Rptd'], inplace=True)

    # Mocodes is list of codes, find its length and first code as two separate features
    df['Mocodes len'] = df['Mocodes'].transform(lambda x: len(str(x).split()))
    df['Mocodes first'] = df['Mocodes'].transform(lambda x: str(x).split()[0]).astype('category')
    df.drop(columns='Mocodes', inplace=True)

            
    # 4. compute class label
    
    # | Status | Status Desc            | 猜测含义                       |   数量 |
    # | :----- | :--------------------- | :----------------------------- | -----: |
    # | IC     | Invert Cont （默认值） | 还在侦查阶段，还没破案         | 830261 |
    # | AA     | Adult Arrest           | 已破案，罪犯是成年人且已逮捕   | 121561 |
    # | AO     | Adult Other            | 已破案，罪犯是成年人且...      | 111185 |
    # | JA     | Juv Arrest             | 已破案，罪犯是未成年人且已逮捕 |   8760 |
    # | JO     | Juv Other              | 已破案，罪犯是未成年人且...    |   2734 |
    # | CC     | UNK                    | 未知                           |     19 |
    # | TH     | UNK                    | 未知                           |      1 |
    # | 13     | UNK                    | 未知                           |      1 |

    # 去除未知的CC, TH, 13
    # status_todrop = ['CC', 'TH', '13']
    # df = df[df['Status'].transform(lambda x: False if x in status_todrop else True)]

    # drop rows with missing 'Status'
    df = df[df['Status'].notna()]

    # 将未破案的为一类IC
    # 已破案的为一类AA, AO, JA, JO,
    y_col = 'Status'
    df[y_col] = df[y_col].transform(lambda y: 1 if y=='IC' else 0)


    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # 'DR_NO' has unique value for each row, 去除id列
        df.drop(columns='DR_NO', inplace=True)
        
        # The following pair of columns are one-to-one mapping, drop less informative column
        # 'AREA' and 'AREA NAME'
        # 'Crm Cd' and 'Crm Cd Desc' , 'Crm Cd 1'  
        # 'Premis Cd', 'Premis Desc' (both 803 and 805 are mapped to 'RETIRED (DUPLICATE) DO NOT USE THIS CODE')
        # 'Weapon Used Cd', 'Weapon Desc',  222 -> np.nan, rest is one-to-one
        # 'Status', 'Status Desc',  nan, 13, TH, CC -> UNK, rest is one-to-one (involve only a few records) 
        df.drop(columns=['AREA', 'Crm Cd', 'Crm Cd 1', 'Premis Cd', 'Weapon Used Cd', 'Status Desc'], inplace=True)

        # drop text columns, that need more sophisticated feature extraction
        df.drop(columns=cols_text, inplace=True)
        
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)


    # 6. sampling by class
    # if select_year is not None:
    #     # year    samples
    #     # 2010    208895
    #     # 2012    201231
    #     # 2011    200486
    #     # 2014    195110
    #     # 2013    192280
    #     # 2015     76519    
    #     df = df[df['Year OCC'] == select_year]
    
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)
    
        if drop_useless:
            useless_cols_dict = util.find_useless_colum(df, min_rows_per_value=10)
            df = util.drop_useless(df, useless_cols_dict, verbose=verbose)
    

    return df, y_col



if __name__ == "__main__":
    df,y_col = load(verbose=True)
    df.info()