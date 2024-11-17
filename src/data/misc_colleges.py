# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:49:52 2021
contains information about U.S. colleges and schools
downloaded from https://beachpartyserver.azurewebsites.net/VueBigData/DataFiles/Colleges.txt.

@author: iwenc, QiuRunwen
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load(data_dir="../../data", drop_useless=True, num_sample=None, verbose=False):
    # 1. read/uncompress data
    file_name = os.path.join(data_dir, "misc/Colleges/Colleges.txt")
    df = pd.read_csv(
        file_name,
        sep="\t",
        encoding="latin1",
        na_values=["", "PrivacySuppressed"],
        index_col=0,
    )

    # column                        dtype       na  physcial_type  desc
    # UNITID                        int64           id
    # School Name                   category
    # City                          category
    # State                         category
    # ZIP                           category
    # School Webpage                category    ''
    # Latitude                      float64     ''  GPS position
    # Longitude                     float64     ''  GPS position
    # Admission Rate                float64	    ''  rate[0,1.0]
    # SAT Verbal Midrange           float64     ''  uint16
    # SAT Math Midrange             float64     ''  uint16
    # SAT Writing Midrange          float64     ''  uint16
    # ACT Combined Midrange         float64     ''  uint8
    # ACT English Midrange          float64     ''  uint8
    # ACT Math Midrange             float64     ''  uint8
    # ACT Writing Midrange          float64     ''  uint8
    # SAT Total Average             float64     ''  uint16
    # Undergrad Size                float64     ''  uint32
    # Percent White                 float64     ''  rate[0,1.0]
    # Percent Black                 float64     ''  rate[0,1.0]
    # Percent Hispanic              float64     ''  rate[0,1.0]
    # Percent Asian                 float64     ''  rate[0,1.0]
    # Percent Part Time             float64     ''  rate[0,1.0]
    # Average Cost Academic Year    float64     ''  money
    # Average Cost Program Year     float64     ''  money
    # Tuition (Instate)             float64     ''  money
    # Tuition (Out of state)        float64     ''  money
    # Spend per student             float64     ''  money
    # Faculty Salary                float64     ''  money
    # Percent Part Time Faculty     float64     ''  rate[0,1.0]
    # Percent Pell Grant            float64     ''  rate[0,1.0]
    # Completion Rate               float64     ''  rate[0,1.0]
    # Average Age of Entry          float64     ''  rate[0,1.0]
    # Percent Married               float64     ''  rate[0,1.0]
    # Percent Veteran               float64     ''  rate[0,1.0]
    # Predominant Degree            category    'None'
    # Highest Degree                category
    # Ownership                     category
    # Region                        category
    # Gender                        category
    # Carnegie Basic Classification category    ''
    # Carnegie Undergraduate        category    ''
    # Carnegie Size                 category    ''
    # Religious Affiliation         category    ''
    # Percent Female                float64     '', 'PrivacySuppressed' rate[0,1.0]
    # agege24                       float64     '', 'PrivacySuppressed' rate[0,1.0]
    # faminc                        float64     '', 'PrivacySuppressed' rate[0,1.0]
    # Mean Earnings 6 years         float64     '', 'PrivacySuppressed' rate[0,1.0]
    # Median Earnings 6 years       float64     '', 'PrivacySuppressed' rate[0,1.0]
    # Mean Earnings 10 years        float64     '', 'PrivacySuppressed' rate[0,1.0]
    # Median Earnings 10 years      float64     '', 'PrivacySuppressed' rate[0,1.0]

    # 2. convert numeric/categorical columns
    cat_cols = [
        "School Name",
        "City",
        "State",
        "ZIP",
        "School Webpage",
        "Predominant Degree",
        "Highest Degree",
        "Ownership",
        "Region",
        "Gender",
        "Carnegie Basic Classification",
        "Carnegie Undergraduate",
        "Carnegie Size",
        "Religious Affiliation",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    #
    # convert into binary classification, 1 if `Mean Earnings 6 years' > 30000
    y_col = "Mean Earnings 6 years"
    df = df[df[y_col].notna()]

    # y_col = 'MeanEarning6Year>=30000'
    # df[y_col] = df['Mean Earnings 6 years'].apply(lambda x: 1 if x>=30000 else 0)
    # df.drop(columns='Mean Earnings 6 years', inplace=True)

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        # drop id column, unique value for each record
        df.drop(columns="UNITID", inplace=True)

        # earning over 10 years leaks information when predicting earnings over 6 years, they should be deleted
        df.drop(
            columns=[
                "Median Earnings 6 years",
                "Mean Earnings 10 years",
                "Median Earnings 10 years",
            ],
            inplace=True,
        )

        # b. auto identified useless columns
        # drop empty columns: ['Average Age of Entry', 'Percent Married', 'Percent Veteran']
        # drop columns with more than 50% missing: ['Admission Rate', 'SAT Verbal Midrange', 'SAT Math Midrange', 'SAT Writing Midrange', 'ACT Combined Midrange', 'ACT English Midrange', 'ACT Math Midrange', 'ACT Writing Midrange', 'SAT Total Average', 'Average Cost Program Year', 'Completion Rate', 'Carnegie Undergraduate', 'Carnegie Size', 'Religious Affiliation']
        # drop columns with average sample per category < 2: ['School Name', 'ZIP', 'School Webpage']
        # drop columns with a single category more than 90% samples: ['Gender']
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
    assert df[y_col].isna().sum() == 0
