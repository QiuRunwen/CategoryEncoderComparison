# -*- coding: utf-8 -*-
"""
CPMP-2015-regression
it is a benchmark result of the Container Pre-Marshalling Problem (CPMP).
The target is to predict the runtime of a CPMP algorithm on a given instance.

https://www.openml.org/d/41700

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

    file_path = os.path.join(data_dir, r"OpenML\CPMP2015", "CPMP2015.csv")
    df = pd.read_csv(file_path)

    # 2108*27
    # Feature Name	Type	Distinct/Missing Values
    # runtime (target)	numeric	1161 distinct values 0 missing attributes
    # instance_id	string	527 distinct values 0 missing attributes
    # repetition	numeric	1 distinct values 0 missing attributes
    # stacks	numeric	10 distinct values 0 missing attributes
    # tiers	numeric	4 distinct values 0 missing attributes
    # stack.tier.ratio	numeric	17 distinct values 0 missing attributes
    # container.density	numeric	7 distinct values 0 missing attributes
    # empty.stack.pct	numeric	15 distinct values 0 missing attributes
    # overstowing.stack.pct	numeric	22 distinct values 0 missing attributes
    # overstowing.2cont.stack.pct	numeric	34 distinct values 0 missing attributes
    # group.same.min	numeric	2 distinct values 0 missing attributes
    # group.same.max	numeric	12 distinct values 0 missing attributes
    # group.same.mean	numeric	13 distinct values 0 missing attributes
    # group.same.stdev	numeric	139 distinct values 0 missing attributes
    # top.good.min	numeric	8 distinct values 0 missing attributes
    # top.good.max	numeric	31 distinct values 0 missing attributes
    # top.good.mean	numeric	281 distinct values 0 missing attributes
    # top.good.stdev	numeric	484 distinct values 0 missing attributes
    # overstowage.pct	numeric	69 distinct values 0 missing attributes
    # bflb	numeric	49 distinct values 0 missing attributes
    # left.density	numeric	215 distinct values 0 missing attributes
    # tier.weighted.groups	numeric	522 distinct values 0 missing attributes
    # avg.l1.top.left.lg.group	numeric	218 distinct values 0 missing attributes
    # cont.empty.grt.estack	numeric	62 distinct values 0 missing attributes
    # pct.bottom.pct.on.top	numeric	22 distinct values 0 missing attributes
    # algorithm	nominal	4 distinct values 0 missing attributes
    # runstatus	nominal	3 distinct values 0 missing attributes

    # 2. convert numeric/categorical columns
    cat_cols = ["instance_id", "algorithm", "runstatus"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. regression target
    y_col = "runtime"
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
