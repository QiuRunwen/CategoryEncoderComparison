# -*- coding: utf-8 -*-
"""
One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that 
the vehicle might have serious issues that prevent it from being sold to customers. 
The auto community calls these unfortunate purchases "kicks".

https://www.openml.org/d/41162

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

    filepath = os.path.join(data_dir, "OpenML", "kick", "kick.csv")
    with open(filepath) as f:
        df = pd.read_csv(f)

    # 6590*9

    # 2. convert numeric/categorical columns

    # Feature Name	Type	Distinct/Missing Values
    # IsBadBuy (target)	nominal	2 distinct values   0 missing attributes
    # PurchDate	numeric	517 distinct values    0 missing attributes
    # Auction	nominal	3 distinct values    0 missing attributes
    # VehYear	numeric	10 distinct values    0 missing attributes
    # VehicleAge	numeric	10 distinct values    0 missing attributes
    # Make	nominal	33 distinct values    0 missing attributes
    # Model	nominal	1063 distinct values    0 missing attributes
    # Trim	nominal	134 distinct values    2360 missing attributes
    # SubModel	nominal	863 distinct values    8 missing attributes
    # Color	nominal	16 distinct values    8 missing attributes
    # Transmission	nominal	3 distinct values    9 missing attributes
    # WheelTypeID	nominal	4 distinct values    3169 missing attributes
    # WheelType	nominal	3 distinct values    3174 missing attributes
    # VehOdo	numeric	39947 distinct values    0 missing attributes
    # Nationality	nominal	4 distinct values    5 missing attributes
    # Size	nominal	12 distinct values    5 missing attributes
    # TopThreeAmericanName	nominal	4 distinct values    5 missing attributes
    # MMRAcquisitionAuctionAveragePrice	numeric	10342 distinct values    18 missing attributes
    # MMRAcquisitionAuctionCleanPrice	numeric	11379 distinct values    18 missing attributes
    # MMRAcquisitionRetailAveragePrice	numeric	12725 distinct values    18 missing attributes
    # MMRAcquisitonRetailCleanPrice	numeric	13456 distinct values    18 missing attributes
    # MMRCurrentAuctionAveragePrice	numeric	10315 distinct values    315 missing attributes
    # MMRCurrentAuctionCleanPrice	numeric	11265 distinct values    315 missing attributes
    # MMRCurrentRetailAveragePrice	numeric	12493 distinct values    315 missing attributes
    # MMRCurrentRetailCleanPrice	numeric	13192 distinct values    315 missing attributes
    # PRIMEUNIT	nominal	2 distinct values    69564 missing attributes
    # AUCGUART	nominal	2 distinct values    69564 missing attributes
    # BYRNO	nominal	74 distinct values    0 missing attributes
    # VNZIP1	nominal	153 distinct values    0 missing attributes
    # VNST	nominal	37 distinct values    0 missing attributes
    # VehBCost	numeric	2010 distinct values    68 missing attributes
    # IsOnlineSale	nominal	2 distinct values    0 missing attributes
    # WarrantyCost	numeric	281 distinct values    0 missing attributes

    cat_cols = [
        "Auction",
        "Make",
        "Model",
        "Trim",
        "SubModel",
        "Color",
        "Transmission",
        "WheelTypeID",
        "WheelType",
        "Nationality",
        "Size",
        "TopThreeAmericanName",
        "PRIMEUNIT",
        "AUCGUART",
        "BYRNO",
        "VNZIP1",
        "VNST",
        "IsOnlineSale",
    ]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction
    # df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit="s")

    # 4. compute class label
    #     IsBadBuy
    # 0    64007
    # 1     8976

    y_col = "IsBadBuy"
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
