"""
Plot for the result.
"""
from typing import Callable
from types import MappingProxyType
import os
import shutil
import itertools
import warnings
import pandas as pd
import numpy as np

# import scipy
from scipy import stats
from scipy import optimize

# import statsmodels
import scikit_posthocs as sp

# from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn import metrics, manifold, preprocessing, cluster, decomposition, impute
from sklearn.ensemble import RandomForestRegressor
from adjustText import adjust_text

import utils
from models import CLASSIFIER_NAMES

USE_LATEX = True
if USE_LATEX:
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]}
    )

RESULT_DIR = "../output/result/current"
# OUTPUT_DIR = os.path.join(RESULT_DIR, 'plot')
PATH_DATA_DESC = "../output/data_desc/alldata_desc.csv"

DICT_ECD_RENAME = MappingProxyType(
    {
        "Target": "SShrink",
        "LeaveOneOutSigma": "LeaveOneOut",
        "BackwardDifference": "BackDiff",
    }
)

ENCODERS = (
    # "Delete",
    "OneHot",
    "BaseN",
    "BackDiff",
    "Helmert",
    "Sum",
    "Similarity",
    "MinHash",
    "Ordinal",
    "Count",
    "Mean",
    "SShrink",
    "MEstimate",
    # 'Quantile',
    # 'JamesSteinBinary',
    "JamesStein",
    # 'CatBoost',
    # 'LeaveOneOut',
    # 'LeaveOneOutSigma',
    "GLMM",
    # 'WOE',
    # 'Auto',
    # 'Average',
)

DICT_ENCODER_DIMENSION1 = MappingProxyType(
    {
        "Delete": " - ",
        "Drop": " - ",
        "OneHot": " + ",
        "BaseN": " + ",
        "Binary": " + ",
        "Dummy": " + ",
        "Helmert": " + ",
        "BackwardDifference": " + ",
        "BackDiff": " + ",
        "Sum": " + ",
        "Polynomial": " + ",
        "Ordinal": " = ",
        "Count": " = ",
        "Hashing": " + ",
        "Similarity": " + ",
        "MinHash": " + ",
        "Mean": " = ",
        "Target": " = ",
        "SShrink": " = ",
        "JamesStein": " = ",
        "JamesSteinBinary": " = ",
        "Quantile": " = ",
        "Summary": " + ",
        "MEstimate": " = ",
        "GLMM": " = ",
        "WOE": " = ",
        "LeaveOneOut": " = ",
        "LeaveOneOutSigma": " = ",
        "CatBoost": " = ",
    }
)

DICT_ENCODER_DIMENSION2 = MappingProxyType(
    {
        "Delete": "0",
        "Drop": "0",
        "OneHot": "c",
        "BaseN": "log c",
        "Binary": "log c",
        "Dummy": "c-1",
        "Helmert": "c-1",
        "BackwardDifference": "c-1",
        "BackDiff": "c-1",
        "Sum": "c-1",
        "Polynomial": "c-1",
        "Ordinal": "1",
        "Count": "1",
        "Similarity": "c",
        "MinHash": "30",
        "Mean": "1",
        "Target": "1",
        "SShrink": "1",
        "JamesStein": "1",
        "JamesSteinBinary": "1",
        "Quantile": "1",
        "Summary": "1",
        "MEstimate": "1",
        "GLMM": "1",
        "WOE": "1",
        "LeaveOneOut": "1",
        "LeaveOneOutSigma": "1",
        "CatBoost": "1",
    }
)

DICT_MODEL_RENAME = MappingProxyType(
    {
        "XGB": "XGBoost",
        "DecisionTree": "DT",
        "LinearSVR": "SVM",
        "LinearSVC": "SVM",
        "RidgeCV": "LNR",
        "LogisticRegression": "LGR",
        "RandomForest": "RF",
        "LGBM": "LightGBM",
        "MLP": "NN",
    }
)

MODELS = (
    "LGR",
    "LNR",
    "NN",
    "SVM",
    "DT",
    "RF",
    "XGBoost",
    "LightGBM",
)

DICT_DATASET_RENAME = MappingProxyType({})

DSS = (
    "Obesity",
    "CPMP2015",
    "TripAdvisor",
    "Autism",
    "Moneyball",
    "Socmob",
    "Cholesterol",
    "GermanCredit",
    "Mammographic",
    "Wholesale",
    "Avocado",
    "CPS1988",
    #    'KDDCup09', # TOO SLOW
    "EmployeeAccess",
    #    'AutoLoan', # TOO SLOW
    "Kick",
    "Churn",
    "HIV",
    "RoadSafety",
    "CarEvaluation",
    "Mushroom",
    "Adult",
    "Nursery",
    "Colleges",
    "EmployeeSalaries",
    "StudentPerformance",
    # 'NycTaxi', # TOO SLOW
    "HousingPrice",
    "BikeSharing",
    "Diamonds",
    "UkAir",
    # 'Average'
)
DSS_HIGH = (
    "Obesity",
    "CPMP2015",
    "TripAdvisor",
    "Autism",
    "Moneyball",
    "Socmob",
    "Cholesterol",
    "Colleges",
    # "KDDCup09",
    "EmployeeAccess",
    "EmployeeSalaries",
    # "AutoLoan",
    "Kick",
    "Churn",
)  # sample_per_level < 100


DSS_NORMAL = tuple(ds for ds in DSS if ds not in DSS_HIGH)

MEASURES_REG = ("MSE", "MAE", "RMSE", "MAPE")
MEASURES_CLS = (
    "F1",
    "AUC",
    "Accuracy",
    "Recall",
    "Aver_precision",
    "Balanced_accuracy",
    "CrossEntropy",
    "ZeroOneLoss",
    "HingeLoss",
)

DICT_MODEL_TYPE = MappingProxyType(
    {
        "LR": "ATIM",
        "LNR": "ATIM",
        "LGR": "ATIM",
        "NN": "ATIM",
        "SVM": "ATIM",
        "DT": "TB-F",
        "RF": "TB-F",
        "XGBoost": "TB-D",
        "LightGBM": "TB-D",
        "CatBoost": "TB-D",
        "Average": "Average",
    }
)

DICT_MODEL_TYPE2 = MappingProxyType(
    {
        "LR": "ATIM",
        "LNR": "ATIM",
        "LGR": "ATIM",
        "NN": "ATIM",
        "SVM": "ATIM",
        "DT": "TM",
        "RF": "TM",
        "XGBoost": "TM",
        "LightGBM": "TM",
        "CatBoost": "TM",
        "Average": "Average",
    }
)


def is_task_regression(measure: str) -> bool:
    if measure in MEASURES_REG:
        return True
    elif measure in MEASURES_CLS:
        return False
    else:
        raise ValueError(f"Unknown measure: {measure}")


def output_excel(
    df: pd.DataFrame,
    filepath: str,
    func_name_style: Callable[[pd.Series], pd.Series] = None,
    axis=0,
    sheet_name="Sheet1",
):
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)

    fp_exists = os.path.exists(filepath)
    if fp_exists:
        # keep the original sheet
        dict_name_df = pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
        # https://github.com/pylint-dev/pylint/issues/3060
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")  # pylint: disable=abstract-class-instantiated
        for _sheet_name, df_sheet in dict_name_df.items():
            df_sheet.to_excel(writer, sheet_name=_sheet_name, index=False)
    else:
        dict_name_df = {sheet_name: df}
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")  # pylint: disable=abstract-class-instantiated

    # write the dataframe to the Excel file
    if func_name_style is not None:
        df_style = df.style.apply(func_name_style, axis=axis)
        df_style.to_excel(writer, sheet_name=sheet_name)
    else:
        df.to_excel(writer, sheet_name=sheet_name)

    for _sheet_name, worksheet in writer.sheets.items():
        # get the worksheet object
        # worksheet = writer.sheets[_sheet_name]
        df_sheet = dict_name_df.get(_sheet_name, df)

        # adjust column width
        df_tmp = (
            df_sheet.reset_index() if "Unnamed: 0" not in df_sheet.columns else df_sheet
        )
        for i, col in enumerate(df_tmp.columns):
            column_len = max(df_tmp[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)

    # save the Excel file
    writer.close()


def filt_and_sort(df: pd.DataFrame, models=None, encoders=None, dss=None):
    """df, models, encoders, dss = filt_and_sort(df, models, encoders, dss)"""
    if models is not None:
        df = df[df["model"].isin(models)]
        tmp_models = df["model"].unique()
        models = [model for model in models if model in tmp_models]

    if encoders is not None:
        df = df[df["encoder"].isin(encoders)]
        tmp_ecds = df["encoder"].unique()
        encoders = [ecd for ecd in encoders if ecd in tmp_ecds]

    if dss is not None:
        df = df[df["dataset"].isin(dss)]
        tmp_dss = df["dataset"].unique()
        dss = [ds for ds in dss if ds in tmp_dss]
    return df.copy(), models, encoders, dss


def read_and_filter(
    result_dir,
    d_ecd_rename: dict[str, str] = DICT_ECD_RENAME,
    d_model_rename: dict[str, str] = DICT_MODEL_RENAME,
    d_ds_rename: dict[str, str] = DICT_DATASET_RENAME,
    encoders: list[str] = ENCODERS,
    models: list[str] = MODELS,
    dss: list[str] = DSS,
    check=True,
    handle_delete=False,
) -> pd.DataFrame:
    filepath = os.path.join(result_dir, "result.csv")
    df = pd.read_csv(filepath)

    # TODO carefully check the result
    # GLMM can't be used to train model when the dataset has too many rows.
    # currently, KDDCup09 is not used to train GLMM
    # cols = df.columns
    # dict_record = MappingProxyType(dict(zip(cols, [np.nan]*len(cols))))
    # records = []
    # df_kddcup_glmm = df[(df['dataset']=='KDDCup09') & (df['encoder']=='GLMMEncoder')]
    # for seed in df['seed'].unique():
    #     if seed in df_kddcup_glmm['seed'].unique():
    #         continue
    #     for model in df['model'].unique():
    #         if model not in CLASSIFIER_NAMES+["MLP2Classifier", "MLPClassifier_400"]: # TODO name is not the same as func_name
    #             continue
    #         for ds in ['KDDCup09']:
    #             tmp_dict = dict_record.copy()
    #             tmp_dict['exp_id'] = f"{seed}_{ds}_GLMMEncoder_{model}"
    #             tmp_dict["seed"] = seed
    #             tmp_dict["model"] = model
    #             tmp_dict["encoder"] = 'GLMMEncoder'
    #             tmp_dict["dataset"] = ds
    #             tmp_dict["is_task_reg"] = False # TODO currently, the one happend to be classification task
    #             records.append(tmp_dict)
    # df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)

    df = df.sort_values(["seed", "dataset", "encoder", "model"]).reset_index(drop=True)
    df["encoder"] = df["encoder"].str.replace("Encoder", "")
    df["model"] = df["model"].str.replace(r"Classifier|Regressor", "", regex=True)

    # preprocess
    d_ecd_rename = DICT_ECD_RENAME if d_ecd_rename is None else d_ecd_rename
    d_model_rename = DICT_MODEL_RENAME if d_model_rename is None else d_model_rename
    d_ds_rename = DICT_DATASET_RENAME if d_ds_rename is None else d_ds_rename
    df["encoder"] = df["encoder"].transform(lambda ecd: d_ecd_rename.get(ecd, ecd))
    df["model"] = df["model"].transform(lambda model: d_model_rename.get(model, model))
    df["dataset"] = df["dataset"].transform(lambda ds: d_ds_rename.get(ds, ds))

    df = df[df["encoder"].isin(encoders)] if encoders is not None else df
    df = df[df["model"].isin(models)] if models is not None else df
    df = df[df["dataset"].isin(dss)] if dss is not None else df

    # TODO
    # When using Delete encoder, the dataset that only has categorical features can't be use to train model.
    # Currently, we only use target to predict. Binary: Majority, Regression: Mean.
    if handle_delete:
        df.loc[df["train_dim"] == 0, MEASURES_REG + MEASURES_CLS] = np.nan

    if check:
        check_result(df, num_fold=10, dss=dss, models=models, encoders=encoders)
        # print(df["dataset"].value_counts())
        # print(df["encoder"].value_counts())
        # print(df["model"].value_counts())

    return df


def check_result(
    df: pd.DataFrame, num_fold=10, dss=None, models=None, encoders=None, check_glmm=True
):
    # TODO check Auto
    df = df[df["encoder"] != "Auto"]  # Auto is not a encoder

    dss = DSS if dss is None else dss
    models = MODELS if models is None else models
    encoders = ENCODERS if encoders is None else encoders

    dss = [ds for ds in dss if ds != "Average"]
    assert set(df["dataset"].unique()) == set(dss), "DSS not match"

    # check classification dataset
    df_cls = df[~df["is_task_reg"]]

    _models = [m for m in models if m != "LNR"]
    assert set(df_cls["model"].unique()) == set(_models), "cls: MODELS not match"
    assert (
        df_cls["model"].value_counts().nunique() == 1
    ), "cls: MODELS have different num"
    _encoders = [ecd for ecd in encoders if ecd not in ("Quantile", "Auto", "Average")]
    assert set(df_cls["encoder"].unique()) == set(_encoders), "cls: ENCODERS not match"
    assert (
        df_cls["encoder"].value_counts().nunique() == 1
    ), "cls: ENCODERS have different num"
    assert (
        df_cls.groupby(["dataset", "model", "encoder"])["seed"].count() == num_fold
    ).all(), "num_fold not match"

    # check regression dataset
    df_reg = df[df["is_task_reg"]]
    _models = [m for m in models if m != "LGR"]
    assert set(df_reg["model"].unique()) == set(_models), "reg: MODELS not match"
    assert (
        df_reg["model"].value_counts().nunique() == 1
    ), "reg: MODELS have different num"
    _encoders = [ecd for ecd in encoders if ecd not in ("WOE", "Auto", "Average")]
    assert set(df_reg["encoder"].unique()) == set(_encoders), "reg: ENCODERS not match"
    assert (
        df_reg["encoder"].value_counts().nunique() == 1
    ), "reg: ENCODERS have different num"
    assert (
        df_reg.groupby(["dataset", "model", "encoder"])["seed"].count() == num_fold
    ).all(), "num_fold not match"

    # if check_glmm:
    #     df_glmm = df[df['encoder']=='GLMM']
    #     if df_glmm["seed"].nunique() != num_fold:
    #         warnings.warn(f"All num_fold: {num_fold}. But GLMM num_fold: {df_glmm['seed'].nunique()}")
    #     if df_glmm["dataset"].nunique() != len(dss):
    #         warnings.warn(f"All num_dss: {len(dss)}. But GLMM num_dss: {df_glmm['dataset'].nunique()}")
    #     if df_glmm["model"].nunique() != len(models):
    #         warnings.warn(f"All num_models: {len(models)}. But GLMM num_models: {df_glmm['model'].nunique()}")


def is_measure_larger_better(measure: str) -> bool:
    if measure in (
        "F1",
        "AUC",
        "Accuracy",
        "Recall",
        "Aver_precision",
        "Balanced_accuracy",
    ):
        return True
    elif measure in (
        "CrossEntropy",
        "ZeroOneLoss",
        "HingeLoss",
        "MSE",
        "MAE",
        "RMSE",
        "MAPE",
        "train_time(s)",
    ):
        return False
    else:
        raise ValueError(f"Unknown measure: {measure}")


def draw_box(
    df: pd.DataFrame,
    ds: str,
    measure: str,
    models: list[str] = None,
    encoders: list[str] = None,
    figsize: tuple[float, float] = None,
    output_dir: list[str] = None,
    showfliers=True,
    y_is_model=False,
    save_pdf=False,
    use_rank=False,
    add_aspl=False,
    show_title=True,
):
    df = df[df["dataset"] == ds]
    df, models, encoders, _ = filt_and_sort(df, models, encoders)

    if use_rank:
        df = calc_scenario_statistic(
            df,
            measure=measure,
            models=models,
            encoders=encoders,
            use_mean=False,
            rank_method="average",
        )
        measure = "rank"

    if figsize is None:
        figsize = (2 + 2 * len(models), 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if y_is_model:
        # g = sns.FacetGrid(df, row="model", palette="Set2", sharey=False)
        # g.map_dataframe(sns.boxplot, y=measure, x="encoder", showfliers=showfliers)

        # rotate xticklabels
        # for ax in g.axes.ravel():
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        g = sns.boxplot(
            data=df,
            x="model",
            y=measure,
            hue="encoder",
            order=models,
            hue_order=encoders,
            ax=ax,
            orient="v",
            showfliers=showfliers,  # 是否显示异常值
        )
    else:
        if len(models) == 1:
            g = sns.boxplot(
                data=df,
                x="encoder",
                y=measure,
                order=encoders,
                ax=ax,
                showfliers=showfliers,  # 是否显示异常值
            )
        else:
            g = sns.boxplot(
                data=df,
                x="model",
                y=measure,
                hue="encoder",
                order=models,
                hue_order=encoders,
                ax=ax,
                showfliers=showfliers,  # 是否显示异常值
            )

    title = ds
    if len(models) != 1:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.get_legend().set_title("encoder")
    else:
        # rotate xticklabels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        title = f"{title}, {models[0]}"

    # Draw Shadow
    col_len = len(models)
    if y_is_model or col_len == 1:
        pass
        # ymin, ymax = ax.get_ylim()
        # interval = (ymax - ymin)/col_len
        # a,b = divmod(col_len,2) # 计算列是奇数还是偶数
        # color_ls = ['#FFFFFF','#F5F5F5']*a # 颜色等价于['white','whitesmoke']
        # if b: # 如果是1，则列数是奇数，需要再加多一列
        #     color_ls.append('#FFFFFF')
        # colormap = mpl.colors.ListedColormap(color_ls)

        # ax.pcolorfast(ax.get_xlim(), np.arange(ymin,ymax+ interval, interval),
        #     np.arange(col_len).reshape(-1,col_len),
        #     cmap=colormap)
        # ax.set_ylim(ymin,ymax) # 重新设置回之前的大小
    else:
        xmin, xmax = ax.get_xlim()
        interval = (xmax - xmin) / col_len
        a, b = divmod(col_len, 2)  # 计算列是奇数还是偶数
        color_ls = ["#FFFFFF", "#F5F5F5"] * a  # 颜色等价于['white','whitesmoke']
        if b:  # 如果是1，则列数是奇数，需要再加多一列
            color_ls.append("#FFFFFF")
        colormap = mpl.colors.ListedColormap(color_ls)

        ax.pcolorfast(
            np.arange(xmin, xmax + interval, interval),
            ax.get_ylim(),
            np.arange(col_len).reshape(-1, col_len),
            cmap=colormap,
        )

        ax.set_xlim(xmin, xmax)  # 重新设置回之前的大小

    if show_title:
        ax.set_title(title)
    if output_dir is not None:
        data_desc_path = PATH_DATA_DESC
        fn_tmp = ""
        if add_aspl:
            df_data_desc = pd.read_csv(data_desc_path)
            aspl = int(df_data_desc.set_index("name")["sample_per_level"][ds])
            title += f", aspl:{aspl}"
            fn_tmp = f"_aspl={aspl}"
        os.makedirs(output_dir, exist_ok=True)
        fn = f"{measure}{fn_tmp}_{ds}_" + "_".join(models)
        path = os.path.join(output_dir, fn + ".png")
        plt.savefig(path, bbox_inches="tight")
        if save_pdf:
            path_pdf = os.path.join(output_dir, fn + ".pdf")
            plt.savefig(path_pdf, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()


def draw_box_all(
    df: pd.DataFrame,
    measure="F1",
    use_relative: bool = False,
    baseline: str = None,
    models=None,
    encoders=None,
    dss=None,
    output_dir=None,
    # draw_min_s_per_level=True,
    scale_time=True,
    col_wrap=2,
    output_pdf: bool = False,
    fig_size: tuple[float, float] = None,
    showfliers=True,
):
    df = df[df[measure].notna()]
    larger_is_better = is_measure_larger_better(measure)
    if use_relative:
        df = df.reset_index(drop=True)

        def _f(df: pd.DataFrame):
            ser = df[measure]
            baseline_val = ser.max() if larger_is_better else ser.min()
            if baseline is not None:
                baseline_val = df.set_index("encoder").loc[baseline, measure]
            ser = (ser / baseline_val - 1).abs()
            return ser

        tmp_measure = "relative_gap_" + measure
        df[tmp_measure] = (
            df.groupby(["dataset", "model", "seed"], as_index=False)
            .apply(_f)
            .reset_index(0, drop=True)
        )
        measure = tmp_measure

        # print(df[((df['encoder']=='OneHot') == (df[measure]==1))][['exp_key',measure]])
        # print(df[df['encoder']=='OneHot'][measure]!=1)

    # df = df.rename(columns={measure:measure})

    df, models, encoders, dss = filt_and_sort(df, models, encoders, dss)

    # ser_dataset_min_s = pd.Series(dtype='int')
    # fn_data_desc = '../myarticle/resource/Data Desription.xlsx'
    # if draw_min_s_per_level and os.path.exists(fn_data_desc):
    #     ser_dataset_min_s = pd.read_excel(fn_data_desc,sheet_name=1).set_index('Dataset')['Minimum of Average sample per level']
    #     df['dataset'] = df['dataset'].transform(lambda dataset: f'{dataset}, min_sample_per_level:{ser_dataset_min_s[dataset]}')

    tmp_dict = {model: i for i, model in enumerate(models)}
    df = df.sort_values("model", key=lambda x: x.map(tmp_dict))
    g: sns.FacetGrid = sns.catplot(
        data=df,
        x="model",
        y=measure,
        hue="encoder",
        kind="box",
        col="dataset",
        col_wrap=col_wrap,
        sharey=False,
        hue_order=encoders,
        col_order=dss,
        whis=10 if use_relative else 1.5,  # 1.5
        #       height=5, aspect=2
        showfliers=showfliers,  # 是否显示异常值
    )

    # https://seaborn.pydata.org/generated/seaborn.FacetGrid.__init__.html
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    if fig_size is not None:
        g.fig.set_size_inches(*fig_size)

    for ax in g.axes:
        col_len = df["model"].unique().shape[0]
        xmin, xmax = ax.get_xlim()

        interval = (xmax - xmin) / col_len

        a, b = divmod(col_len, 2)  # 计算列是奇数还是偶数
        color_ls = ["#FFFFFF", "#F5F5F5"] * a  # 颜色等价于['white','whitesmoke']
        if b:  # 如果是1，则列数是奇数，需要再加多一列
            color_ls.append("#FFFFFF")
        colormap = mpl.colors.ListedColormap(color_ls)

        if measure == "train_time(s)" and scale_time:
            ax.set_yscale("log")
        else:
            ymin, ymax = ax.get_ylim()
            if use_relative:
                # ymin, ymax = (0.96, 1.005) if larger_is_better else (0.995, 1.04)
                ymin, ymax = (-0.005, 0.05)
                # if baseline is not None:
                #     ax.axhline(1, color='black')
                # pass
            # else:
            #     ymin,ymax = ax.get_ylim()
            #     if larger_is_better:
            #         ymin = ymax*0.95
            #     else:
            #         ymax = ymin*1.05
            ax.set_ylim(ymin, ymax)

        ax.pcolorfast(
            np.arange(xmin, xmax + interval, interval),
            ax.get_ylim(),
            np.arange(col_len).reshape(-1, col_len),
            cmap=colormap,
        )

        ax.set_xlim(xmin, xmax)  # 重新设置回之前的大小

    if output_dir is None:
        plt.show()
    else:
        os.makedirs(output_dir, exist_ok=True)
        fn = f"{baseline}baseline_" if use_relative and baseline is not None else ""
        fn += measure + "_" + "_".join(models)
        if output_pdf:
            path = os.path.join(output_dir, fn + ".pdf")
            plt.savefig(path, bbox_inches="tight")
        path = os.path.join(output_dir, fn + ".png")
        plt.savefig(path, bbox_inches="tight")
        plt.close("all")


def calc_scenario_statistic(
    df: pd.DataFrame,
    measure,
    models=None,
    dss=None,
    encoders=None,
    use_mean=True,
    rank_method="average",
    data_desc_path=PATH_DATA_DESC,
    add_imp_from_res=False,
):
    """
    `rank_method='average'` : [1.1, 1.1, 2.0] -> [1.5, 1.5, 3.0]. `min`: [1.1, 1.1, 2.0] -> [1.0, 1.0, 3.0].
    `dense`: [1.1, 1.1, 2.0] -> [1.0, 1.0, 2.0]. `ordinal`: [1.1, 1.1, 2.0] -> [1.0, 2.0, 3.0].
    """
    df = df[df[measure].notna()]
    df, models, _, dss = filt_and_sort(df, models=models, dss=dss, encoders=encoders)
    df["time"] = df["time_encoder"] + df["time_model"]
    if use_mean:
        df = (
            df.groupby(["dataset", "model", "encoder"], as_index=False)[
                [measure, "train_size", "train_dim", "time"]
            ]
            .mean()
            .dropna()
        )
    else:
        df = df[
            [
                "seed",
                "dataset",
                "model",
                "encoder",
                measure,
                "train_size",
                "train_dim",
                "time",
            ]
        ]

    def group_func(df: pd.DataFrame):
        df["sort_key"] = (
            -df[measure] if is_measure_larger_better(measure) else df[measure]
        )
        # df = df.sort_values(measure, ascending=is_measure_larger_better(measure))
        df["score"] = df[measure]
        df["rank"] = stats.rankdata(
            df["sort_key"], method=rank_method
        )  # [1.1, 2, 3, 1.1] -> [1.5, 3, 4, 1.5]
        df["Avg"] = df[measure].mean()
        df["Best"] = df[df["rank"] == df["rank"].min()][measure].values[0]
        df["gap2best"] = ((df["Best"] - df[measure]) / df["Best"]).abs()
        df["gap2avg"] = (
            (df["Avg"] - df[measure]) / df["Avg"]
            if is_measure_larger_better(measure)
            else (df[measure] - df["Avg"]) / df["Avg"]
        )
        return df

    if use_mean:
        df_rank = (
            df.groupby(["dataset", "model"], as_index=False)
            .apply(group_func)
            .reset_index(drop=True)
        )
    else:
        df_rank = (
            df.groupby(["seed", "dataset", "model"], as_index=False)
            .apply(group_func)
            .reset_index(drop=True)
        )

    if data_desc_path is not None:
        df_alldata_desc = pd.read_csv(data_desc_path, index_col="name")
        df_alldata_desc = df_alldata_desc[
            [
                "row",
                "num_col",
                "cat_col",
                "max_card",
                "sum_card",
                "major_class_ratio",
                "imp_of_max_card_col",
                "sum_cat_importance",
                "sample_per_level",
                "isTaskReg",
            ]
        ]

        df_alldata_desc["sample_per_sumcard"] = (
            df_alldata_desc["row"] / df_alldata_desc["sum_card"]
        )
        df_alldata_desc["volume"] = df_alldata_desc["row"] * df_alldata_desc["sum_card"]
        tmp_df = df_rank["dataset"].apply(lambda x: df_alldata_desc.loc[x])
        df_rank = pd.concat([df_rank, tmp_df], axis=1)

    if add_imp_from_res:
        df_dataset_imp = _calc_imp_from_res(df, measure)
        tmp_df = df_rank["dataset"].apply(lambda x: df_dataset_imp.loc[x])
        df_rank = pd.concat([df_rank, tmp_df], axis=1)

    return df_rank


def _calc_imp_from_res(df: pd.DataFrame, measure: str):
    larger_is_better = is_measure_larger_better(measure)

    def f(_df: pd.DataFrame):
        idx_best = _df[measure].idxmax() if larger_is_better else _df[measure].idxmin()
        score_best = _df.loc[idx_best, measure]
        encoder_best = _df.loc[idx_best, "encoder"]
        if encoder_best == "Delete":
            warnings.warn(f'Delete in {_df["dataset"].iloc[0]} is the best')
        model_best = _df.loc[idx_best, "model"]
        ser_best_model = _df[_df["model"] == model_best].set_index("encoder")[measure]

        # TODO gap2best is not very good
        if "Delete" in ser_best_model.index and pd.notna(ser_best_model["Delete"]):
            sum_cat_imp = abs((ser_best_model["Delete"] - score_best) / score_best)
        else:
            warnings.warn(f'No Delete in {_df["dataset"].iloc[0]}. Use sum_cat_imp=1')
            sum_cat_imp = 1

        # if sum_cat_imp > 1:
        #     sum_cat_imp = 1
        #     warnings.warn(f'sum_cat_imp > 1 in {_df["dataset"].iloc[0]} and set to 1')

        return pd.Series(
            {
                "sum_cat_imp_from_res": sum_cat_imp,
                "best_in_ds": f"{encoder_best}-{model_best}",
            }
        )

    df_mean = (
        df.groupby(["dataset", "model", "encoder"], as_index=False)[measure]
        .mean()
        .dropna()
    )
    df_dataset_imp = df_mean.groupby("dataset", as_index=True).apply(f)
    return df_dataset_imp


def plot_statistic_scatter(
    df: pd.DataFrame,
    measures,
    ecd="OneHot",
    statistic_x="sample_per_level",
    statistic_y="gap2best",
    data_desc_path=PATH_DATA_DESC,
    scalex=False,
    scaley=False,
    move_legend=False,
    plot_regline=False,
    models=None,
    dss=None,
    encoders=None,
    save_path=None,
    renamex=None,
    renamey=None,
    percenty=False,
    fontsize=None,
    rotation=None,
    figsize=None,
    fontsizex=None,
    fontsizey=None,
):
    """
    statistic: 'row','num_col','cat_col','max_card','sum_card','major_class_ratio', 'imp_of_max_card_col','sample_per_level'
    'sum_cat_importance', 'sample_per_sumcard', "train_size","train_dim" 'sample_per_dim', 'volume'
    """
    ls_df_ss = []
    for measure in measures:
        _df = df[df[measure].notna()]
        _df, _, _, _ = filt_and_sort(_df, models, encoders, dss)
        _df_ss = calc_scenario_statistic(
            _df, measure, models, dss, encoders, data_desc_path=data_desc_path
        )
        _df_ss["sample_per_dim"] = _df_ss["train_size"] / _df_ss["train_dim"]
        if _df_ss.empty:
            warnings.warn(f"No data for measure={measure}")
            continue
        ls_df_ss.append(_df_ss)
    df_ss = pd.concat(ls_df_ss)

    # set the size of font
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    fig, ax = plt.subplots(figsize=figsize)
    if fontsizex is not None:  # set xlabel size
        ax.xaxis.label.set_size(fontsizex)
    if fontsizey is not None:
        ax.yaxis.label.set_size(fontsizey)
    if ecd is not None:
        df_ss_ecd = df_ss[df_ss["encoder"] == ecd].copy()
        # df_ss_ecd['gap2best'] = df_ss_ecd['gap2best'].abs()
        df_ss_ecd["Model"] = df_ss_ecd["model"]
        sns.scatterplot(
            data=df_ss_ecd,
            x=statistic_x,
            y=statistic_y,
            ax=ax,
            hue="Model",
            hue_order=models,
            style="Model",
            markers=True,
        )
    else:
        # plot all encoders
        # TODO not implemented completely. Only for `time`and it's a boxplot
        df_ss = df_ss[df_ss["encoder"] != "Delete"]
        df_ss_ecd = df_ss.copy()
        # set the order of encoder according to the average of 'Time'
        tmp = (
            df_ss_ecd.groupby("encoder", as_index=False)["time"]
            .mean()
            .sort_values("time")
        )
        encoders_ordered = tmp["encoder"].tolist()

        ############line plot############
        df_ss["encoder"] = df_ss["encoder"].map(
            lambda x: f"{x}({DICT_ENCODER_DIMENSION2[x]})"
        )
        hue_order = [
            f"{ecd}({DICT_ENCODER_DIMENSION2[ecd]})" for ecd in encoders_ordered[::-1]
        ]
        sns.lineplot(
            data=df_ss,
            x="train_size",
            y=statistic_y,
            ax=ax,
            hue="encoder",
            hue_order=hue_order,
            style="encoder",
            markers=True,
            dashes=False,
            errorbar=None,
        )
        # make legend outside the plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.get_legend().set_title("Encoder")

        ############box plot############
        # sns.boxenplot(data=df_ss, x='encoder',order=encoders_ordered, y=statistic_y, ax=ax)
        # pos_lines = [['SShrink', 'BaseN'],['MinHash', 'Sum'], ['BackDiff', 'GLMM'],
        #              ['GLMM', 'OneHot'], ['Similarity', 'MinHash']]
        # # draw 2 vertical lines to separate the boxes
        # for i in range(0, len(encoders_ordered)-1):
        #     if encoders_ordered[i:i+2] in pos_lines:
        #         ax.axvline(i+0.5, linestyle='--', color='red')
        # # rename the xticklabels
        # current_xticklabels = [text.get_text() for text in ax.get_xticklabels()]
        # new_xticklabels = [f'{ecd}({DICT_ENCODER_DIMENSION2[ecd]})' for ecd in current_xticklabels]
        # ax.set_xticklabels(new_xticklabels, rotation=rotation)

        ############scatter plot############
        # ax = sns.scatterplot(data=df_ss_ecd, x="encoder", y=statistic_y,
        # hue='model', hue_order=models, style='model', markers=True)
        # df_ss_ecd = df_ss.groupby(['encoder', statistic_x], as_index=False)[statistic_y].max()
        # ax = sns.lineplot(data=df_ss_ecd, x=statistic_x, y=statistic_y,
        # hue='encoder', hue_order=encoders, style='encoder', markers=True)
        # ax = sns.scatterplot(data=df_ss, x=statistic_x, y=statistic_y,
        # hue='encoder', hue_order=encoders, style='model', markers=True)
    if plot_regline:
        # sns.regplot(data=df_ss_ecd, x=statistic_x, y=statistic_y, scatter=False, ax=ax)
        x = df_ss_ecd[statistic_x]
        y = df_ss_ecd[statistic_y]
        a, b = optimize.curve_fit(lambda t, a, b: a * np.exp(b * t), x, y, p0=(1, -1))[
            0
        ]
        sns.lineplot(x=x, y=a * np.exp(b * x), ax=ax, color="red")

    # set the position of legend
    if move_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # set the scale of x and y axis
    if scalex:
        ax.set_xscale("log")
    if scaley:
        ax.set_yscale("log")

    # rename x and y axis
    if renamex is not None:
        ax.set_xlabel(renamex)
    if renamey is not None:
        ax.set_ylabel(renamey)

    # set the format of y axis
    if percenty:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for path in utils.convert_pathoffig(save_path):
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    # reset the size of font
    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})

    return df_ss_ecd


def calc_gap2best_rank(
    df, measures, models=None, dss=None, encoders=None, dict_model_type=None
) -> dict[str, pd.DataFrame]:
    drop_delete = True
    if drop_delete:
        df = df[df["encoder"] != "Delete"]

    ls_df_ss = []
    for measure in measures:
        df_tmp = df[df[measure].notna()]
        df_tmp, models_tmp, encoders_tmp, dss_tmp = filt_and_sort(
            df_tmp, models=models, dss=dss, encoders=encoders
        )
        df_ss_tmp = calc_scenario_statistic(
            df_tmp,
            measure,
            models_tmp,
            dss_tmp,
            encoders_tmp,
            use_mean=True,
            rank_method="average",
        )
        ls_df_ss.append(df_ss_tmp)
    df_ss = pd.concat(ls_df_ss)

    if dict_model_type is None:
        df_ss["model_type"] = df_ss["model"]
    else:
        df_ss["model_type"] = df_ss["model"].map(dict_model_type)

    d_m_res = {}
    for model_type, df_modeltype in df_ss.groupby("model_type"):
        avg_gap2best = df_modeltype.groupby("encoder", as_index=True)["gap2best"].mean()
        std_gap2best = df_modeltype.groupby("encoder", as_index=True)["gap2best"].std()
        gap2best_avgstd = (
            avg_gap2best.map(lambda x: f"{x*100:5.2f}")
            + "±"
            + std_gap2best.map(lambda x: f"{x*100:5.2f}")
        )
        df_res = pd.DataFrame(
            {
                "avg_gap2best": avg_gap2best,
                "std_gap2best": std_gap2best,
                "gap2best_avgstd": gap2best_avgstd,
            },
            index=avg_gap2best.index,
        )
        df_res.sort_values("avg_gap2best", inplace=True)
        df_res["rank"] = stats.rankdata(df_res["avg_gap2best"], method="average")
        d_m_res[model_type] = df_res

    return d_m_res


def output_gap2best_rank(
    df, dss, dss_high, encoders, models, output_dir, measures=("F1", "MSE")
):
    measure_cls, measure_reg = measures
    df = df.copy()
    dss_normal = [ds for ds in dss if ds not in dss_high]
    dict_model_type = DICT_MODEL_TYPE2  # DICT_MODEL_TYPE
    tmp_output_dir = os.path.join(output_dir, "gap2best")
    os.makedirs(tmp_output_dir, exist_ok=True)
    df["model_type"] = df["model"].map(lambda x: dict_model_type[x])
    for dss_desc, _dss in zip(["normal", "high", "allds"], [dss_normal, dss_high, dss]):
        for task_desc, _measures in zip(
            ["classification", "regression", "alltask"],
            [[measure_cls], [measure_reg], measures],
        ):
            _encoders = list(encoders)
            _encoders.remove("Delete") if "Delete" in _encoders else None
            if task_desc == "alltask":
                # WOE only for classification
                # Quantile only for regression
                _encoders.remove("WOE") if "WOE" in _encoders else None
                _encoders.remove("Quantile") if "Quantile" in _encoders else None

            for d_m_type, path_suffix in zip(
                [None, dict_model_type, {m: "all" for m in models}],
                ["", "_type", "_type"],
            ):
                dict_model_dfres = calc_gap2best_rank(
                    df[df["dataset"].isin(_dss)],
                    measures=_measures,
                    models=models,
                    dss=_dss,
                    encoders=_encoders,
                    dict_model_type=d_m_type,
                )
                for model, df_res in dict_model_dfres.items():
                    df_res.to_csv(
                        os.path.join(
                            tmp_output_dir,
                            f"{dss_desc}_{task_desc}_{model}{path_suffix}.csv",
                        ),
                        index=True,
                    )

    print("-" * 5, "output_excel", "-" * 5)

    def func_bold_best(ser: pd.Series):
        ser_format = pd.Series("", index=ser.index)
        ser_format[ser.idxmin()] = "font-weight:bold"
        return ser_format

    modeltypes = list(models) + ["ATIM_type", "TM_type", "all_type"]
    for dss_desc, task_desc in itertools.product(
        ["normal", "high", "allds"], ["classification", "regression", "alltask"]
    ):
        fp_save = os.path.join(
            tmp_output_dir + "_excel", f"{dss_desc}_{task_desc}.xlsx"
        )
        dfs = []
        dfs_avgstd = []
        for model in modeltypes:
            fp = os.path.join(tmp_output_dir, f"{dss_desc}_{task_desc}_{model}.csv")
            if (model == "LNR" and task_desc == "classification") or (
                model == "LGR" and task_desc == "regression"
            ):
                # LNR only for regression, LGR only for classification
                print(f"skip {fp}")
                continue
            df_res = pd.read_csv(fp)
            # df_res_type = pd.read_csv(os.path.join(tmp_output_dir, f"{dss_desc}_{task_desc}_{model}_type.csv"))
            # df_res = pd.concat([df_res, df_res_type])
            df_res = df_res.sort_values(by="avg_gap2best")
            # df_res = df_res.rename(columns={'encoder':'Encoder','avg_rank':'Average Rank','pvalue':'p-value'})
            # df_res = df_res[['Encoder','Average Rank','p-value']]
            output_excel(df_res, filepath=fp_save, sheet_name=model)

            df_tmp = df_res[["encoder", "avg_gap2best"]].copy()
            df_tmp.rename(columns={"avg_gap2best": model}, inplace=True)
            df_tmp.set_index("encoder", inplace=True)
            dfs.append(df_tmp)

            df_tmp = df_res[["encoder", "gap2best_avgstd"]].copy()
            df_tmp.rename(columns={"gap2best_avgstd": model}, inplace=True)
            df_tmp.set_index("encoder", inplace=True)
            dfs_avgstd.append(df_tmp)

        df_summary_avg = pd.concat(dfs, axis=1)
        df_summary_avgstd = pd.concat(dfs_avgstd, axis=1)

        output_excel(df_summary_avgstd, filepath=fp_save, sheet_name="summaryAvgStd")
        output_excel(
            df_summary_avg,
            func_name_style=func_bold_best,
            filepath=fp_save,
            sheet_name="summary",
        )


def output_time_lineplot(
    df: pd.DataFrame,
    ls_models=(("NN", "SVM", "RF", "XGBoost"),),
    dss=("EmployeeAccess", "UkAir"),
    figsize=(10, 4),
    fontsize=18,
    rotation=45,
    errorbar=("ci", 95),
    save_dir=None,
):
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    df = df.copy()
    df["time"] = df["time_encoder"] + df["time_model"]
    for models, ds in itertools.product(ls_models, dss):
        df_time = df[(df["dataset"] == ds) & df["model"].isin(models)].copy()
        ecd_order = df_time.groupby("encoder")["train_dim"].mean().rank(method="first")
        df_time["order"] = df_time["encoder"].map(ecd_order)
        fig, ax = plt.subplots(figsize=figsize)

        palette = None
        # if len(models)==4:
        # The first two similar colors, and the last two similar colors
        # palette = sns.color_palette(["crimson", "deeppink", "dodgerblue", "blue"])
        # palette = sns.color_palette(["#ed7d31", "#f7cbac", "#4472c4", "#b4c6e7"])

        sns.lineplot(
            df_time,
            x="order",
            y="time",
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            hue_order=models,
            ax=ax,
            errorbar=errorbar,
            palette=palette,
        )
        x_tick_positions = ecd_order.to_numpy()
        x_labels = [f"{ecd}({DICT_ENCODER_DIMENSION2[ecd]})" for ecd in ecd_order.index]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_labels, rotation=rotation)

        ax.set_xlabel("Encoders")
        ax.set_ylabel("Time(s)")
        ax.get_legend().set_title("Models")
        ax.set_yscale("log")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"time_{ds}_{'_'.join(models)}.png")
            for path in utils.convert_pathoffig(save_path):
                plt.savefig(path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def output_final():
    result_dir = RESULT_DIR
    output_dir = os.path.join(result_dir, "final")
    os.makedirs(output_dir, exist_ok=True)
    encoders = ENCODERS
    models = ("LNR", "LGR", "NN", "SVM", "DT", "RF", "XGBoost", "LightGBM")
    dss = tuple(ds for ds in DSS if ds not in ["NycTaxi"])
    measures = ("F1", "RMSE")

    dss_high = DSS_HIGH  # sample_per_level < 100
    df = read_and_filter(
        result_dir, encoders=encoders, models=models, dss=dss, check=True
    )

    print("-" * 5, "output_time_lineplot", "-" * 5)
    output_time_lineplot(
        df,
        ls_models=(("NN", "SVM", "RF", "XGBoost"),),
        dss=("UkAir", "EmployeeAccess"),
        figsize=(10, 4),
        fontsize=18,
        rotation=75,
        errorbar=("ci", 95),
        save_dir=output_dir,
    )

    print("-" * 5, "plot_statistic_scatter", "-" * 5)
    plot_statistic_scatter(
        df,
        measures,
        ecd="OneHot",
        statistic_x="sample_per_level",
        statistic_y="gap2best",
        # statistic_y='rank', # 不好看
        data_desc_path=PATH_DATA_DESC,
        scalex=True,
        models=["LGR", "LNR", "NN", "SVM"],
        dss=None,
        encoders=None,
        save_path=os.path.join(output_dir, "spl_gap2best_scatter.png"),
        renamex="minASPL",
        renamey="Relative performance difference",
        # scaley=True,
        percenty=True,
        fontsize=20,
        fontsizey=16,
    )

    plot_statistic_scatter(
        df,
        measures,
        ecd="Mean",
        statistic_x="sample_per_level",
        statistic_y="gap2best",
        data_desc_path=PATH_DATA_DESC,
        scalex=True,
        models=["DT", "RF", "XGBoost", "LightGBM"],
        dss=None,
        encoders=None,
        save_path=os.path.join(output_dir, "spl_gap2best_scatter_TM_Mean.png"),
        renamex="minASPL",
        renamey="Relative performance difference",
        # scaley=True,
        percenty=True,
        fontsize=20,
        fontsizey=16,
    )

    print("-" * 5, "output_gap2best_rank", "-" * 5)
    output_gap2best_rank(
        df, dss, dss_high, encoders, models, output_dir, measures=measures
    )

    print("-" * 5, "calc_scenario_statistic", "-" * 5)
    df_ss1 = calc_scenario_statistic(
        df,
        measure=measures[0],
        models=models,
        dss=dss,
        encoders=encoders,
        use_mean=True,
        rank_method="average",
    )
    df_ss2 = calc_scenario_statistic(
        df,
        measure=measures[1],
        models=models,
        dss=dss,
        encoders=encoders,
        use_mean=True,
        rank_method="average",
    )
    df_ss = pd.concat([df_ss1, df_ss2])
    df_ss.to_csv(os.path.join(output_dir, "scenario_statistic.csv"), index=False)

    def tmp_func(_df: pd.DataFrame):
        _df["tmprank"] = stats.rankdata(_df["sort_key"], method="min")
        return _df[_df["tmprank"] == _df["tmprank"].min()][["model", "encoder"]]

    df_dataset_bestmodelencoder = (
        df_ss.groupby("dataset").apply(tmp_func).reset_index().drop(columns=["level_1"])
    )
    df_dataset_bestmodelencoder.sort_values("dataset").to_csv(
        os.path.join(output_dir, "dataset_bestmodelencoder.csv")
    )
    df_dataset_bestmodelencoder[
        ["model", "encoder"]
    ].value_counts().reset_index().pivot(
        index="encoder", columns="model", values="count"
    ).to_csv(
        os.path.join(output_dir, "best_model_encoder.csv")
    )


if __name__ == "__main__":
    output_final()
