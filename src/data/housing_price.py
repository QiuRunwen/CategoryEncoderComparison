"""
California Housing Prices
Median house prices for California districts derived from the 1990 census.

https://www.kaggle.com/datasets/camnugent/california-housing-prices

author: QiuRunwen
"""

import pandas as pd
import os

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_path = os.path.join(data_dir, "kaggle/housing_price", "housing.csv")
    df = pd.read_csv(file_path)
    # RangeIndex: 20640 entries, 0 to 20639
    # Data columns (total 10 columns):
    # #   Column              Non-Null Count  Dtype
    # ---  ------              --------------  -----
    # 0   longitude           20640 non-null  float64
    # 1   latitude            20640 non-null  float64
    # 2   housing_median_age  20640 non-null  float64
    # 3   total_rooms         20640 non-null  float64
    # 4   total_bedrooms      20433 non-null  float64
    # 5   population          20640 non-null  float64
    # 6   households          20640 non-null  float64
    # 7   median_income       20640 non-null  float64
    # 8   median_house_value  20640 non-null  float64
    # 9   ocean_proximity     20640 non-null  category  'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'

    # 2. convert numeric/categorical columns
    cat_cols = ["ocean_proximity"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute target
    y_col = "median_house_value"

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
