# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:00:29 2021

@author: iwenc, RunwenQiu
"""

import logging
import os
import warnings
import re
from pathlib import Path
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import utils
import data

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
SAVE_DIR = os.path.join(ROOT_DIR, "output", "datasets")
DATA_LOADER_DICT = MappingProxyType(
    {
        "CPS1988": data.cps1988.load,
        # "ChscaseFoot": chscase_foot.load, # 只有3个值，应该是多分类
        "Socmob": data.socmob.load,
        "Moneyball": data.moneyball.load,
        "Avocado": data.avocado.load,
        "CPMP2015": data.cpmp2015.load,
        "Cholesterol": data.cholesterol.load,
        "Wholesale": data.wholesale.load,
        "TripAdvisor": data.trip_advisor.load,
        "Autism": data.autism.load,
        "Mammographic": data.mammographic.load,
        "Obesity": data.obesity.load,
        "GermanCredit": data.german_credit.load,
        "Adult": data.uci_adult.load,
        "Colleges": data.misc_colleges.load,
        "AutoLoan": data.tianchi_auto_loan_default_risk.load,
        "Crime": data.lacity_crime.load,
        "EmployeeSalaries": data.employee_salaries.load,
        "H1BVisa": data.h1b_visa.load,
        # 'LendingClub': data.lending_club.load,
        "TrafficViolations": data.traffic_violation.load,
        "RoadSafety": data.road_safety_accident.load,
        "CarEvaluation": data.car_evaluation.load,
        "CatInDat": data.kaggle_cat_in_dat.load2,
        "HousingPrice": data.housing_price.load,
        "StudentPerformance": data.student_performance.load,
        "Diamonds": data.diamonds.load,
        "NycTaxi": data.nyc_taxi.load,
        "UkAir": data.ukair.load,
        "BikeSharing": data.bike_sharing.load,
        "Mushroom": data.mushroom.load,
        "Nursery": data.nursery.load,
        "EmployeeAccess": data.employee_access.load,
        "KDDCup09": data.kdd_cup09.load,
        "HIV": data.hiv.load,
        "Kick": data.kick.load,
        "Churn": data.churn.load,
    }
)


def get_data_loader():
    # Define a dictionary of data sets
    # Key is the name of a data set
    # Value is a function that load data set
    return DATA_LOADER_DICT


def get_default_name_kwargs(sort=False):
    """Get the default name and kwargs of all datasets.
    sort: whether sort the result by the number of rows in ascending order.
    """
    data_loader_dict = get_data_loader()
    res = {}
    for func_name in data_loader_dict:
        res[func_name] = {
            "func_name": func_name,
            "kwargs": {},
        }

    if sort:
        data_desc_path = os.path.join(ROOT_DIR, "output/data_desc/alldata_desc.csv")
        if os.path.exists(data_desc_path):
            df = pd.read_csv(data_desc_path)
            df.sort_values("row", inplace=True)
            res = {name: res[name] for name in df["name"]}
        else:
            warnings.warn(f"{data_desc_path} does not exist. Use default order.")

    return res


def get_data_desc(
    df: pd.DataFrame,
    y_col: str = None,
    high_card_col_imp: bool = False,
    imp_output_dir: str = None,
    use_cache: bool = True,
):
    X = df if y_col is None else df[[col for col in df.columns if col != y_col]]
    cat_cols = X.columns[X.dtypes == "category"]

    major_class_ratio = None
    pos_class_ratio = None
    isTaskReg = None
    if y_col is not None:
        ser_class_count = df[y_col].value_counts()
        isTaskReg = ser_class_count.shape[0] != 2
        if ser_class_count.shape[0] == 2:  # only support binaray classification
            major_class_ratio = (ser_class_count / df.shape[0]).iloc[0]
            pos_class_ratio = (ser_class_count / df.shape[0]).loc[1]

    ser_col_card = calc_col_card(X) # only calculate the cardinality of X
    res_dict = {
        "row": X.shape[0],
        "num_col": X.shape[1] - cat_cols.size,
        "cat_col": cat_cols.size,
        "max_card": ser_col_card.iloc[0],
        "sum_card": ser_col_card.fillna(1).sum(),
        "major_class_ratio": major_class_ratio,
        "pos_class_ratio": pos_class_ratio,
        "isTaskReg": isTaskReg,
    }
    df_col_card_importance = ser_col_card.to_frame("cardinality")
    if high_card_col_imp:
        y = df[y_col]
        ser_col_importance = calc_col_importance(
            X, y, save_dir=imp_output_dir, use_cache=use_cache
        )
        df_col_card_importance["importance"] = ser_col_importance
        max_card_col = ser_col_card.index[0]
        res_dict["max_card_col"] = max_card_col
        res_dict["imp_of_max_card_col"] = ser_col_importance.loc[max_card_col]
        res_dict["sum_cat_importance"] = ser_col_importance[cat_cols].sum()

    if imp_output_dir is not None:
        os.makedirs(imp_output_dir, exist_ok=True)
        df_col_card_importance.to_csv(
            os.path.join(imp_output_dir, "col_card_importance.csv")
        )

    return res_dict


def calc_col_card(df: pd.DataFrame, y_col: str = None):
    """Calculate the cardinality of each column in a dataset.
    For categorical feature, the cardinality is its number of unique categories (NAN also count).
    For numerical feature, the cardinality is np.nan for compatibility and caculation of effective cardinality of a dataset.

    Args:
        df (pd.DataFrame): A dataset with correct dtype, especially the `category` columns
        y_col (str, optional): _description_. Defaults to None.

    Returns:
        pd.Series: ser_col_card with descending cardinality.
    """
    X = df if y_col is None else df[[col for col in df.columns if col != y_col]]
    cols_cat = X.columns[X.dtypes == "category"]
    # cols_num = [col for col in X.columns if col not in cols_cat]
    ser_col_card = X.apply(
        lambda ser: len(ser.unique()) if ser.name in cols_cat else np.nan
    )

    return ser_col_card.sort_values(ascending=False)


def desc_alldataset(
    data_loader_dict: dict = None,
    data_dir="../data",
    output_dir="../output/data_desc",
    draw_reg=True,
    high_card_col_imp=False,
    use_cache=True,
):
    if data_loader_dict is None:
        data_loader_dict = get_data_loader()
    res_ls = []
    os.makedirs(output_dir, exist_ok=True)
    for name, load_func in data_loader_dict.items():
        print(f"loading `{name}`")
        df, y_col = load_func(data_dir=data_dir)
        output_dir_dataset = os.path.join(output_dir, name)
        output_dir_dataset_valcounts = os.path.join(output_dir_dataset, "valcounts")
        draw_all_cat_hist(df, y_col, output_dir_dataset_valcounts)
        res = get_data_desc(
            df, y_col, high_card_col_imp, output_dir_dataset, use_cache
        )
        res["name"] = name
        res_ls.append(res)
        if draw_reg:
            if res["isTaskReg"]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.lineplot(
                    x=np.arange(df.shape[0]),
                    y=df[y_col].sort_values().values,
                    ax=axes[0],
                )
                sns.histplot(data=df, x=y_col, ax=axes[1])
                fig.suptitle(name)
                fp = os.path.join(output_dir, f"{name}.png")
                fig.savefig(fp)
                plt.close()
                print(
                    f"Target in regression has been drawn and saved in `{os.path.abspath(fp)}`"
                )

    df_res = pd.DataFrame(res_ls)
    df_res["sample_per_level"] = df_res["row"] / df_res["max_card"]

    # Reorder the columns with 'name' in the first position
    df_res = df_res[["name"] + [col for col in df_res.columns if col != "name"]]

    df_res.sort_values("row", inplace=True)
    fp = os.path.join(output_dir, "alldata_desc.csv")
    df_res.to_csv(fp, index=False)
    print(f"The description of all dataset has been saved in `{os.path.abspath(fp)}`")
    print(df_res)
    return df_res


def calc_col_importance(
    X: pd.DataFrame,
    y: pd.Series,
    name: str = "",
    seed=1,
    save_dir="./",
    use_cache=True,
):
    prefix = os.path.join(save_dir, name + "_imp" + "_s=" + str(seed))
    filename = prefix + ".pkl"
    if use_cache and os.path.exists(filename):
        ser_col_importance = utils.load(filename)
    else:
        X = X.copy(deep=True)

        # 1. numerical variable
        #        replace inf by max+1
        #        replace -inf by min-1
        #        filling na by mean (excluding na)
        # 2. encode categorical variable by TargetEncoding
        cat_cols = []
        for col in X:
            if pd.api.types.is_numeric_dtype(X[col]):
                col_max = X[col].max()
                if col_max == np.inf:
                    msg = f"Data set: {name} col: {col} np.inf -> max+1"
                    warnings.warn(msg)
                    new_col = X[col].replace([np.inf], np.nan)
                    col_max = new_col.max()
                    X[col].replace([np.inf], col_max + 1, inplace=True)
                col_min = X[col].min()
                if col_min == -np.inf:
                    msg = f"Data set: {name} col: {col} -np.inf -> min-1"
                    warnings.warn(msg)
                    new_col = X[col].replace([-np.inf], np.nan)
                    col_min = new_col.min()
                    X[col].replace([-np.inf], col_min - 1, inplace=True)

                v = X[col].mean()
                X[col] = X[col].fillna(v)
            elif pd.api.types.is_categorical_dtype(X[col]):
                cat_cols.append(col)
            else:
                warnings.warn(
                    f"col: {col} is not numeric or categorical. it will be ignored."
                )

        y = y.astype("int") if y.nunique() == 2 else y.astype("float")
        X = TargetEncoder(cols=cat_cols).fit_transform(X, y)

        model = (
            RandomForestClassifier(
                random_state=seed,
                n_estimators=100,  # default 100
                criterion="gini",  # default 'gini'
                max_depth=30,  # default None, no limit
                min_samples_split=20,  # default 2
                min_samples_leaf=1,
            )  # default 1
            if y.nunique() == 2
            else RandomForestRegressor(
                random_state=seed,
                n_estimators=100,
                criterion="squared_error",
                max_depth=30,
                min_samples_split=20,
                min_samples_leaf=1,
            )
        )
        model.fit(X, y)

        ser_col_importance = pd.Series(model.feature_importances_, index=X.columns)
        utils.save(ser_col_importance, filename)

    # plot figure
    ser_col_importance = ser_col_importance.sort_values(ascending=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, len(ser_col_importance) / 2))
    ser_col_importance.plot(kind="barh")
    plt.xlabel("variable importance")
    plt.savefig(prefix + ".png")
    plt.close()
    return ser_col_importance


def get_num_cat_cols(df: pd.DataFrame, y_col: str = None):
    """num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols"""
    num_cols = []
    cat_cols = []
    cat_is_num_cols = []
    cat_miss_lable_cols = []
    for col in df.columns:
        if col == y_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]):
            cat_cols.append(col)
            if df[col].astype(str).str.isdigit().all():
                logging.info("dataset:%s, col:%s, 是数字型的分类变量", df.shape, col)
                cat_is_num_cols.append(col)
        else:
            cat_miss_lable_cols.append(col)
            logging.info("dataset:%s, col:%s, dtype不是数字也不是category", df.shape, col)
            cat_cols.append(col)

    return num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols


def draw_all_cat_hist(df, y_col, output_dir, overwrite=False):
    num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols = get_num_cat_cols(
        df, y_col
    )
    for col in tqdm(cat_cols):
        # Define a regular expression pattern to match illegal characters
        illegal_chars_pattern = r'[\\/:\*\?"<>\|]'
        # Replace illegal characters with underscores
        tmp_col = re.sub(illegal_chars_pattern, "_", col)
        fp = os.path.join(output_dir, f"cat_col={tmp_col}.png")
        if os.path.exists(fp) and not overwrite:
            continue
        fig, ax = plt.subplots()
        ser_val_count = df[col].value_counts(sort=True, dropna=False)
        # ser_val_count.plot(kind="barh", ax=ax)
        sns.lineplot(x=np.arange(ser_val_count.size), y=ser_val_count.values, ax=ax)
        fig.suptitle(col)
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(fp)
        plt.close()
        # print(f"cat col `{col}` has been drawn and saved in `{os.path.abspath(fp)}`")


class Dataset:
    __slots__ = [
        "name",
        "df",
        "y_col",
        "X",
        "y",
        "is_task_reg",
        "stats",
        "func_name",
        "kwargs",
    ]

    def __init__(
        self, name: str, df: pd.DataFrame, y_col: str, func_name=None, kwargs=None
    ):
        self.name = name
        self.df = df
        self.y = df[y_col]

        vc = self.y.value_counts(sort=False, dropna=False)
        if vc.index.isna().any():
            raise RuntimeError(f"{name=}, {y_col=} has nan")

        self.is_task_reg = vc.size > 2  # whether binary classification or regression
        if not self.is_task_reg:
            # binary classification
            self.df[y_col] = utils.format_y_binary(self.df[y_col])
            self.y = df[y_col]

        self.X = df.drop(columns=y_col)
        self.y_col = y_col

        self.func_name = func_name
        self.kwargs = kwargs

    def get_largest_card_cat_var(self):
        hc_col = None
        hc_card = -1
        cats = self.X.select_dtypes(include="category")
        for col in cats:
            value_counts = self.X[col].value_counts(sort=False, dropna=False)
            if value_counts.size > hc_card:
                hc_col = col
                hc_card = value_counts.size

        return hc_col

    def rf_importance_series(self, seed=3, save_dir=SAVE_DIR, use_cache=True):
        # plt.show()
        ser_col_importance = calc_col_importance(
            self.X,
            self.y,
            name=self.name,
            seed=seed,
            save_dir=save_dir,
            use_cache=use_cache,
        )
        return ser_col_importance

    def stats_df(self, seed=3, save_dir=SAVE_DIR, use_cache=True, compute_imp=True):
        var2card = {}
        var2avg_samples_per_cat = {}  # cat_variable to samples_per_cat
        cats = self.X.select_dtypes(include="category")
        for col in cats:
            value_counts = self.X[col].value_counts(sort=False, dropna=False)
            var2card[col] = value_counts.size
            var2avg_samples_per_cat[col] = self.X[col].size / value_counts.size

        self.stats = pd.DataFrame(
            data={"cardinality": pd.Series(data=var2card, index=self.X.columns)}
        )
        self.stats["samples_per_cat"] = pd.Series(
            data=var2avg_samples_per_cat, index=self.X.columns
        )

        self.stats["na-ratio"] = self.X.isna().sum() / max(
            self.X.shape[0], 1
        )  # correctly handle empty table

        self.stats["positve_ratio"] = self.y.sum() / self.y.size

        if compute_imp:
            self.stats["importance"] = self.rf_importance_series(
                seed=seed, save_dir=save_dir, use_cache=use_cache
            )
            imp = self.stats["importance"]
            self.stats["importance_rank"] = imp.rank(ascending=False)
            self.stats["importance_rank_cat"] = imp[
                self.stats["cardinality"].isna() == False
            ].rank(ascending=False)

        if save_dir is not None:
            filename = os.path.join(save_dir, self.name + "_s=" + str(seed) + ".csv")
            self.stats.to_csv(filename)
        return self.stats


def load_one_set(
    func_name,
    name=None,
    data_dir=DATA_DIR,
    data_loader_dict=None,
    drop_useless=True,
    cols_drop=None,
    cat_cols_keep=None,
    num_noise_cols_add=None,
    cat_noise_col_card=None,
):
    name = func_name if name is None else name

    if data_loader_dict is None:
        data_loader_dict = get_data_loader()
    load_func = data_loader_dict[func_name]
    df, y_col = load_func(data_dir=data_dir, drop_useless=drop_useless)
    if cols_drop is not None:
        df.drop(columns=cols_drop, inplace=True)

    if cat_cols_keep is not None:
        num_cols, cat_cols, _, _ = get_num_cat_cols(df, y_col)
        df.drop(
            columns=[col for col in cat_cols if col not in cat_cols_keep], inplace=True
        )

    if num_noise_cols_add is not None:
        if not isinstance(num_noise_cols_add, int):
            warnings.warn(f"{num_noise_cols_add=} is not int, try to convert")
            num_noise_cols_add = int(num_noise_cols_add)

        if num_noise_cols_add > 0:
            noise_data = np.random.normal(size=(df.shape[0], num_noise_cols_add))

            noise_df = pd.DataFrame(
                data=noise_data,
                columns=[f"noise_{i}" for i in range(num_noise_cols_add)],
            )
            # df = pd.concat([df, noise_df], axis=1)  # pd.concat will change the dtype and return a copy

            df = pd.merge(df, noise_df, left_index=True, right_index=True, copy=False)

    if cat_noise_col_card is not None:
        if not isinstance(cat_noise_col_card, int):
            warnings.warn(f"{cat_noise_col_card=} is not int, try to convert")
            cat_noise_col_card = int(cat_noise_col_card)

        if cat_noise_col_card > 0:
            noise_data = np.random.randint(0, cat_noise_col_card, size=(df.shape[0], 1))

            df["noise_cat"] = noise_data
            df["noise_cat"] = df["noise_cat"].astype("str").astype("category")

    kwargs = {
        "drop_useless": drop_useless,
        "cols_drop": cols_drop,
        "cat_cols_keep": cat_cols_keep,
        "num_noise_cols_add": num_noise_cols_add,
        "cat_noise_col_card": cat_noise_col_card,
    }
    ds = Dataset(name, df, y_col, func_name, kwargs)
    return ds


def load_all(data_loader_dict=None, drop_useless=True):
    if data_loader_dict is None:
        data_loader_dict = get_data_loader()

    for name in data_loader_dict:
        yield load_one_set(
            name,
            data_dir=DATA_DIR,
            data_loader_dict=data_loader_dict,
            drop_useless=drop_useless,
        )


def output_data(output_dir="../output/data_after_cleaning", data_names=None):
    """Output the data after cleaning, which is used in R for randomForest."""

    data_loader_dict = get_data_loader()
    data_names = data_loader_dict.keys() if data_names is None else data_names
    for name in data_names:
        ds = load_one_set(
            name,
            data_dir="../data",
            data_loader_dict=data_loader_dict,
            drop_useless=False,
        )

        # convert all categorical columns to string
        cat_cols = ds.df.select_dtypes(include="category").columns
        ds.df[cat_cols] = ds.df[cat_cols].astype("str")
        if not ds.is_task_reg:
            ds.df[ds.y_col] = ds.df[ds.y_col].astype("str")

        # put the target column to the first column
        cols = ds.df.columns.tolist()
        cols.remove(ds.y_col)
        cols = [ds.y_col] + cols
        ds.df = ds.df[cols]

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, name + ".csv")
        import csv

        ds.df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"dataset `{name}` has been saved in `{os.path.abspath(filename)}`")


if __name__ == "__main__":
    # output_dir="../output/tmp/data_desc_rf_changed"
    output_dir = "../output/data_desc"
    df = desc_alldataset(
        data_dir="../data",
        output_dir=output_dir,
        draw_reg=True,
        high_card_col_imp=True,
        use_cache=False,
    )

    # output the description of datasets which are used in the paper
    df = pd.read_csv(f"{output_dir}/alldata_desc.csv")
    from analysis_result import output_excel, DSS

    # output_data(output_dir="../output/data_after_cleaning", data_names=DSS)
    datasets = DSS
    # datasets = [
    #     "KDDCup09",
    #     "EmployeeAccess",
    #     "AutoLoan",
    #     "Kick",
    #     "Churn",
    #     "HIV",
    #     "RoadSafety",
    #     "CarEvaluation",
    #     "Mushroom",
    #     "Adult",
    #     "Nursery",
    #     "Colleges",
    #     "EmployeeSalaries",
    #     "StudentPerformance",
    #     "NycTaxi",
    #     "HousingPrice",
    #     "BikeSharing",
    #     "Diamonds",
    #     "UkAir",
    #     "Average",
    # ]
    dict_col_rename = {
        "name": "Name",
        "row": "Row",
        "num_col": "Num.",
        "cat_col": "Cat.",
        "max_card": "Max Card.",
        "sum_card": "Sum Card.",
        "pos_class_ratio": "Ratio",
        "sample_per_level": "row/level",
        "sum_cat_importance": "Sum Imp.",
    }
    columns = list(dict_col_rename.keys())
    df = df[df["name"].isin(datasets)]
    df_output = df.sort_values(["isTaskReg", "sample_per_level"])[columns].rename(
        columns=dict_col_rename
    )

    output_excel(df_output, f"{output_dir}/desc_used_in_paper.xlsx")
