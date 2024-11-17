# -*- coding: utf-8 -*-
"""
This dataset classifies people described by a set of attributes as good or bad credit risks.
700 good and 300 bad credits with 20 predictor variables. Data from 1973 to 1975. 
Stratified sample from actual credits with bad credits heavily oversampled. A cost matrix can be used.

https://archive.ics.uci.edu/dataset/522/south+german+credit
The widely used Statlog German credit data (https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29), 
as of November 2019, suffers from severe errors in the coding information and does not come with any background information. 
The 'South German Credit' data provide a correction and some background information, 
based on the Open Data LMU (2010) representation of the same data and several other German language resources.

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
        data_dir, r"UCI\south+german+credit", "SouthGermanCredit.asc"
    )
    df = pd.read_csv(file_path, sep=" ")

    # 1000*21
    # laufkont    categorical
    # laufzeit    quantitative
    # moral       categorical
    # verw        categorical
    # hoehe       quantitative
    # sparkont    categorical
    # beszeit     quantitative
    # rate        quantitative
    # famges      categorical
    # buerge      categorical
    # wohnzeit    quantitative
    # verm        quantitative
    # alter       quantitative
    # weitkred    categorical
    # wohn        categorical

    # bishkred    binary
    # beruf       binary
    # pers        binary
    # telef       binary
    # gastarb     binary
    # kredit      binary

    # 2. convert numeric/categorical columns
    cat_cols = ["laufkont", "moral", "verw", "sparkont", "famges", "buerge", "weitkred",
                "wohn", "bishkred", "beruf", "pers", "telef", "gastarb"]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # kredit
    # 1    700
    # 0    300

    y_col = "kredit"
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
