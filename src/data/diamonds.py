"""
Diamonds
This classic dataset originally contained the prices and other attributes of almost 54,000 diamonds.
Note: 14184 of those seem to be the same diamonds, measure from a different angle.
This can be found out but checking for duplicated value when disregarding the variables x, y, z ,
depth and table, which are dependent on the angle.

https://www.openml.org/d/44979
"""

import pandas as pd
import os

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_path = os.path.join(data_dir, "OpenML\diamonds", "diamonds.csv")
    df = pd.read_csv(file_path)

    # df.info()
    # RangeIndex: 53940 entries, 0 to 53939
    # Data columns (total 10 columns):
    # #   Column   Non-Null Count  Dtype
    # ---  ------   --------------  -----
    # 0   carat    53940 non-null  float64  weight of the diamond (0.2--5.01)
    # 1   cut      53940 non-null  object   quality of the cut (Fair, Good, Very Good, Premium, Ideal)
    # 2   color    53940 non-null  object   diamond colour, from J (worst) to D (best)
    # 3   clarity  53940 non-null  object   a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
    # 4   depth    53940 non-null  float64  total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
    # 5   table    53940 non-null  float64  width of top of diamond relative to widest point (43--95)
    # 6   x        53940 non-null  float64  length in mm (0--10.74)
    # 7   y        53940 non-null  float64  width in mm (0--58.9)
    # 8   z        53940 non-null  float64  depth in mm (0--31.8)
    # 9   price    53940 non-null  int64    price in US dollars ($326--$18,823)

    # dtypes: int64(16), object(18)

    # 2. convert numeric/categorical columns
    cat_cols = ["cut", "color", "clarity"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute target
    y_col = "price"

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
