"""
UCI Car Evaluation.
Car Evaluation Database was derived from a simple hierarchical decision model originally developed
for the demonstration of a Expert system for decision making. The model proposed should be able to
evaluates cars according to concept structure, i.e. multiple categorical features.

this database may be useful for testing constructive induction and structure discovery methods.
http://archive.ics.uci.edu/ml/datasets/Car+Evaluation

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
    file_path = os.path.join(data_dir, "UCI/CarEvaluation/", "car.data")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
        dtype={
            "buying": "category",
            "maint": "category",
            "doors": "category",
            "persons": "category",
            "lug_boot": "category",
            "safety": "category",
            # 'class':'category',
        },
    )
    # 1728*7
    #    buying       v-high, high, med, low    1728 non-null
    #    maint        v-high, high, med, low    1728 non-null
    #    doors        2, 3, 4, 5-more           1728 non-null
    #    persons      2, 4, more                1728 non-null
    #    lug_boot     small, med, big           1728 non-null
    #    safety       low, med, high            1728 non-null
    # Missing Attribute Values: none
    # Class Distribution (number of instances per class)
    #    class      N          N[%]
    #    -----------------------------
    #    unacc     1210     (70.023 %)
    #    acc        384     (22.222 %)
    #    good        69     ( 3.993 %)
    #    v-good      65     ( 3.762 %)

    # 2. convert numeric/categorical columns
    cat_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    y_col = "class"
    df[y_col] = df[y_col].transform(
        lambda x: 0 if x == "unacc" else 1
    )  # whether accept the car, unaccpet is 0, other is 1.

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
