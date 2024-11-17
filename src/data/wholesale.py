# -*- coding: utf-8 -*-
"""
Wholesale customers
The data set refers to clients of a wholesale distributor.
It includes the annual spending in monetary units (m.u.) on diverse product categories
https://archive.ics.uci.edu/dataset/292/wholesale+customers

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

    file_path = os.path.join(
        data_dir, r"UCI\wholesale+customers", "Wholesale customers data.csv"
    )
    df = pd.read_csv(file_path)

    # 440*8
    # 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
    # 2)	MILK: annual spending (m.u.) on milk products (Continuous);
    # 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
    # 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
    # 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
    # 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
    # 7)	CHANNEL: customersâ€™ Channel - Horeca (Hotel/Restaurant/CafÃ©) or Retail channel (Nominal)
    # 8)	REGION: customersâ€™ Region â€“ Lisnon, Oporto or Other (Nominal)

    # 2. convert numeric/categorical columns
    cat_cols = ["Channel", "Region"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label

    y_col = "Channel"
    # "Horeca": 0, "Retail": 1}
    df[y_col] = df[y_col].map({1: 0, 2: 1})
    assert df[y_col].notna().all()

    # after convert to binary class
    #     Channel
    # 0    298
    # 1    142

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
