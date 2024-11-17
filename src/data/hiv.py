# -*- coding: utf-8 -*-
"""
The data contains lists of octamers (8 amino acids) and a flag (-1 or 1) depending on 
whether HIV-1 protease will cleave in the central position (between amino acids 4 and 5).
https://archive.ics.uci.edu/dataset/330/hiv+1+protease+cleavage
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

    headers = ["amino acids", "cleaved"]

    dfs = []
    for filename in [
        "746Data.txt",
        "1625Data.txt",
        "impensData.txt",
        "schillingData.txt",
    ]:
        filepath = os.path.join(data_dir, "UCI", "hiv+1+protease+cleavage", filename)
        with open(filepath) as f:
            dfs.append(pd.read_csv(f, header=None, names=headers, sep=","))

    df = pd.concat(dfs, ignore_index=True)

    df = pd.concat(
        [
            df,
            df["amino acids"].apply(
                lambda x: pd.Series(list(x), index=[f"aa{i}" for i in range(1, 9)])
            ),
        ],
        axis=1,
    )
    df.drop(columns=["amino acids"], inplace=True)

    # 6590*9

    # 2. convert numeric/categorical columns

    # Each attribute is a letter denoting an amino acid. G (Glycine, Gly); P (Proline, Pro);
    # A (Alanine, Ala); V (Valine, Val); L (Leucine, Leu); I (Isoleucine, Ile); M (Methionine, Met);
    # C (Cysteine, Cys); F (Phenylalanine, Phe); Y (Tyrosine, Tyr); W (Tryptophan, Trp);
    # H (Histidine, His); K (Lysine, Lys); R (Arginine, Arg); Q (Glutamine, Gln); N (Asparagine, Asn);
    # E (Glutamic Acid, Glu); D (Aspartic Acid, Asp); S (Serine, Ser); T (Threonine, Thr).
    # The output denotes cleaved (+1) or not cleaved (-1).

    cat_cols = df.columns
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    #     cleaved
    # 0    5230
    # 1    1360

    y_col = "cleaved"
    df[y_col] = df[y_col].map({-1: 0, 1: 1})
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
