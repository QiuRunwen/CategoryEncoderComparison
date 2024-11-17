"""
nyc-taxi-green-dec-2016
https://www.openml.org/search?type=data&sort=runs&id=42729&status=active

@author: QiuRunwen
"""

import pandas as pd
import os
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 

def load(data_dir='../../data', drop_useless=True, num_sample=None, verbose=False): 
    
    # 1. read/uncompress data
    file_path = os.path.join(data_dir, r'OpenML\nyc-taxi-green-dec-2016','nyc-taxi-green-dec-2016.csv')
    df = pd.read_csv(file_path)
    
    # df.info()
    # 581835*19
    # @ATTRIBUTE VendorID {1, 2}
    # @ATTRIBUTE store_and_fwd_flag {N, Y}
    # @ATTRIBUTE RatecodeID {1, 2, 3, 4, 5}
    # @ATTRIBUTE PULocationID {1, 3, 5, 6, 7,..., 263, 264, 265}
    # @ATTRIBUTE DOLocationID {1, 3, 4, 5, ..., 264, 265}
    # @ATTRIBUTE passenger_count REAL
    # @ATTRIBUTE extra {0, 0.22, 0.5, 1, 4.5}
    # @ATTRIBUTE mta_tax {-0.5, 0, 0.5}
    # @ATTRIBUTE tip_amount REAL
    # @ATTRIBUTE tolls_amount REAL
    # @ATTRIBUTE improvement_surcharge {-0.3, 0, 0.3}
    # @ATTRIBUTE total_amount REAL
    # @ATTRIBUTE trip_type {1, 2}
    # @ATTRIBUTE lpep_pickup_datetime_day INTEGER
    # @ATTRIBUTE lpep_pickup_datetime_hour INTEGER
    # @ATTRIBUTE lpep_pickup_datetime_minute INTEGER
    # @ATTRIBUTE lpep_dropoff_datetime_day INTEGER
    # @ATTRIBUTE lpep_dropoff_datetime_hour INTEGER
    # @ATTRIBUTE lpep_dropoff_datetime_minute INTEGER

    # 2. convert numeric/categorical columns
    cat_cols = ['VendorID', 'store_and_fwd_flag', 'RatecodeID','PULocationID', 'DOLocationID',
                'extra', 'mta_tax', 'improvement_surcharge', 'trip_type']
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')

    # 3. simple feature extraction

    
    # 4. compute target
    y_col = 'tip_amount'

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