"""
Predict student performance in secondary education (high school).
This data approach student achievement in secondary education of two Portuguese schools.
The data attributes include student grades, demographic, social and school related features)
and it was collected by using school reports and questionnaires.
Two datasets are provided regarding the performance in two distinct subjects:
Mathematics (mat) and Portuguese language (por).

https://archive.ics.uci.edu/dataset/320/student+performance

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
    file_path1 = os.path.join(data_dir, "UCI/student_performance", "student-mat.csv")
    file_path2 = os.path.join(data_dir, "UCI/student_performance", "student-por.csv")
    df1 = pd.read_csv(file_path1, encoding="ascii", delimiter=";")
    df2 = pd.read_csv(file_path2, encoding="ascii", delimiter=";")

    df1["subject"] = "Mathematics"
    df2["subject"] = "Portuguese"
    df = pd.concat([df1, df2]).reset_index(drop=True)
    # df.info()

    # RangeIndex: 1044 entries, 0 to 1043
    # Data columns (total 34 columns):
    # 1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
    # 2 sex - student's sex (binary: 'F' - female or 'M' - male)
    # 3 age - student's age (numeric: from 15 to 22)
    # 4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
    # 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
    # 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
    # 7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
    # 8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
    # 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    # 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    # 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
    # 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
    # 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
    # 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    # 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
    # 16 schoolsup - extra educational support (binary: yes or no)
    # 17 famsup - family educational support (binary: yes or no)
    # 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
    # 19 activities - extra-curricular activities (binary: yes or no)
    # 20 nursery - attended nursery school (binary: yes or no)
    # 21 higher - wants to take higher education (binary: yes or no)
    # 22 internet - Internet access at home (binary: yes or no)
    # 23 romantic - with a romantic relationship (binary: yes or no)
    # 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
    # 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
    # 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
    # 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    # 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    # 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
    # 30 absences - number of school absences (numeric: from 0 to 93)

    # these grades are related with the course subject, Math or Portuguese:
    # 31 G1 - first period grade (numeric: from 0 to 20)
    # 31 G2 - second period grade (numeric: from 0 to 20)
    # 32 G3 - final grade (numeric: from 0 to 20, output target)

    # dtypes: int64(16), object(18)

    # 2. convert numeric/categorical columns
    cat_cols = [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "subject",
    ]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute target
    y_col = "G3"

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
