# -*- coding: utf-8 -*-
"""
Avocado Prices

Historical data on avocado prices and sales volume in multiple US markets.
Original source: the Hass Avocado Board website in May of 2018

Downloaded from Kaggle: https://www.kaggle.com/neuromusic/avocado-prices

https://www.openml.org/search?type=data&sort=runs&id=41210&status=active

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

    file_path = os.path.join(data_dir, r"kaggle\avocado", "avocado.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], date_format="%Y-%m-%d")
    df.drop(columns=["Unnamed: 0"], inplace=True)

    # (18249, 13)
    # 'Date', 'AveragePrice', 'Total Volume', '4046', '4225', '4770',
    #    'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year',
    #    'region'
    # Feature Name	Type	Distinct/Missing Values
    # AveragePrice (target)	numeric	259 distinct values 0 missing attributes
    # Date	string	169 distinct values0 missing attributes
    # Total Volume	numeric	18237 distinct values 0 missing attributes
    # 4046	numeric	17702 distinct values 0 missing attributes
    # 4225	numeric	18103 distinct values 0 missing attributes
    # 4770	numeric	12071 distinct values 0 missing attributes
    # Total Bags	numeric	18097 distinct values
    # 0 missing attributes
    
    # Date - The date of the observation
    # AveragePrice - the average price of a single avocado
    # type - conventional or organic
    # Region - the city or region of the observation
    # Total Volume - Total number of avocados sold
    # 4046 - Total number of avocados with PLU 4046 sold
    # 4225 - Total number of avocados with PLU 4225 sold
    # 4770 - Total number of avocados with PLU 4770 sold

    # 2. convert numeric/categorical columns
    cat_cols = ["type", "region"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    # df["weekday"] = df["Date"].dt.weekday # only 1 value
    df.drop(columns=["Date"], inplace=True)

    # 4. regression target
    y_col = "AveragePrice"
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
