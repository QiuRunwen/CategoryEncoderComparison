# -*- coding: utf-8 -*-
"""
Autism Screening Adult
Autistic Spectrum Disorder Screening Data for Adult.
The target is to predict whether a person has Autistic Spectrum Disorder.
https://archive.ics.uci.edu/dataset/426/autism+screening+adult

@author: QiuRunwen
"""

import os
from scipy.io import arff
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data

    file_path = os.path.join(
        data_dir, r"UCI\autism+screening+adult", "Autism-Adult-Data.arff"
    )
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # 704*21
    # Dataset: adult-weka.filters.unsupervised.attribute.NumericToNominal-Rfirst-10
    # A1_Score's type is nominal, range is ('0', '1')
    # A2_Score's type is nominal, range is ('0', '1')
    # A3_Score's type is nominal, range is ('0', '1')
    # A4_Score's type is nominal, range is ('0', '1')
    # A5_Score's type is nominal, range is ('0', '1')
    # A6_Score's type is nominal, range is ('0', '1')
    # A7_Score's type is nominal, range is ('0', '1')
    # A8_Score's type is nominal, range is ('0', '1')
    # A9_Score's type is nominal, range is ('0', '1')
    # A10_Score's type is nominal, range is ('0', '1')
    # age's type is numeric
    # gender's type is nominal, range is ('f', 'm')
    # ethnicity's type is nominal, range is ('White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern ', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'others')
    # jundice's type is nominal, range is ('no', 'yes')
    # austim's type is nominal, range is ('no', 'yes')
    # contry_of_res's type is nominal, range is ('United States', 'Brazil', 'Spain', 'Egypt', 'New Zealand', 'Bahamas', 'Burundi', 'Austria', 'Argentina', 'Jordan', 'Ireland', 'United Arab Emirates', 'Afghanistan', 'Lebanon', 'United Kingdom', 'South Africa', 'Italy', 'Pakistan', 'Bangladesh', 'Chile', 'France', 'China', 'Australia', 'Canada', 'Saudi Arabia', 'Netherlands', 'Romania', 'Sweden', 'Tonga', 'Oman', 'India', 'Philippines', 'Sri Lanka', 'Sierra Leone', 'Ethiopia', 'Viet Nam', 'Iran', 'Costa Rica', 'Germany', 'Mexico', 'Russia', 'Armenia', 'Iceland', 'Nicaragua', 'Hong Kong', 'Japan', 'Ukraine', 'Kazakhstan', 'AmericanSamoa', 'Uruguay', 'Serbia', 'Portugal', 'Malaysia', 'Ecuador', 'Niger', 'Belgium', 'Bolivia', 'Aruba', 'Finland', 'Turkey', 'Nepal', 'Indonesia', 'Angola', 'Azerbaijan', 'Iraq', 'Czech Republic', 'Cyprus')
    # used_app_before's type is nominal, range is ('no', 'yes')
    # result's type is numeric
    # age_desc's type is nominal, range is ('18 and more',)
    # relation's type is nominal, range is ('Self', 'Parent', 'Health care professional', 'Relative', 'Others')
    # Class/ASD's type is nominal, range is ('NO', 'YES')

    # 2. convert numeric/categorical columns

    cat_cols = [col for col in df.columns if col != "age"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # Class/ASD
    # 0    515
    # 1    189

    y_col = "Class/ASD"
    df[y_col] = df[y_col].map({b"NO": 0, b"YES": 1})
    assert df[y_col].notna().all()

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        df.drop(columns=["age_desc"], inplace=True) # only one value

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
