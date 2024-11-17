"""
Bike Sharing Dataset
This dataset contains the hourly and daily count of rental bikes between
years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information.

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Author: QiuRunwen
"""

import os
import pandas as pd


if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_path = os.path.join(data_dir, "UCI/bike_sharing", "hour.csv")
    df = pd.read_csv(
        file_path, index_col="instant", parse_dates=["dteday"], date_format="%Y-%m-%d"
    )

    # df.info()
    # 17389*17
    # - instant: record index
    # - dteday : date
    # - season : season (1:winter, 2:spring, 3:summer, 4:fall)
    # - yr : year (0: 2011, 1:2012)
    # - mnth : month ( 1 to 12)
    # - hr : hour (0 to 23)
    # - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
    # - weekday : day of the week
    # - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
    # + weathersit :
    # 	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    # 	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    # 	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    # 	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    # - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
    # - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
    # - hum: Normalized humidity. The values are divided to 100 (max)
    # - windspeed: Normalized wind speed. The values are divided to 67 (max)
    # - casual: count of casual users
    # - registered: count of registered users
    # - cnt: count of total rental bikes including both casual and registered

    # 2. convert numeric/categorical columns
    cat_cols = ["season", "yr", "mnth", "hr", "holiday", "workingday", "weathersit"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    df.drop(columns="dteday", inplace=True)  # 年月日已经都提取出来了的

    # 4. compute target
    y_col = "cnt"
    df.drop(
        columns=["casual", "registered"], inplace=True
    )  # cnt恒等于casual+registered，需要移除的

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
