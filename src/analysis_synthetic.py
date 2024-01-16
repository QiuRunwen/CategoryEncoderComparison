import itertools
import os
from pathlib import Path
import math
from typing import Callable
import warnings

from tqdm import tqdm
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder, TargetEncoder
from mean_encoder import MeanEncoder
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# from matplotlib.ticker import MaxNLocator, FixedLocator
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from joblib import Parallel, delayed

import utils
from delete_encoder import DeleteEncoder

USE_LATEX = True
if USE_LATEX:
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]}
    )


def gen_reg_with_cat(
    n_sample=100, cardinality=4, noise_std=1, x2_max=None, seed=None, debug=True
):
    r"""Generate a regression dataset with a categorical feature, (a numeric feature) and (a noise).
    The numeric feature is x2, and the noise is epsilon. They are optional when x2_max and noise_std are None.
    .. math::
        y = x_1 + x_2 + \epsilon

    Args:
        n_samples (int, optional): _description_. Defaults to 100.
        cardinality (int, optional): The number of levels of categorical feature. Defaults to 4.
        noise_std (float, optional): The standard deviation of noise. Defaults to 1.
        x2_max (float, optional): x2 in [0, 5). Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, str]: _description_
    """

    assert cardinality >= 2

    seed = np.random.get_state()[1][0] if seed is None else seed
    rng = np.random.default_rng(seed)
    x1 = (
        rng.integers(0, cardinality, size=n_sample)
        if debug
        else rng.integers(1, cardinality + 1, size=n_sample)
    )
    y = x1
    data_dict = {"x1": x1}
    if x2_max is not None:
        x2 = rng.uniform(0, x2_max, size=n_sample)
        y = y + x2  # if use y += x2, y should have changed to float
        data_dict["x2"] = x2
    if noise_std is not None:
        noise = rng.normal(scale=noise_std, size=n_sample)
        y = y + noise
        data_dict["noise"] = noise
    data_dict["y"] = y
    df = pd.DataFrame(data_dict)
    df["x1"] = df["x1"].astype("int").astype("str").astype("category")
    y_col = "y"
    return df, y_col


def exp_reg_with_cat_comp_onehot_phi(
    models=("LinearRegression", "MLPRegressor"),
    n_samples=range(20, 601, 20),
    cardinalitys=(4,),
    noise_stds=(1,),
    x2_maxs=(5,),
    seeds=range(30),
    debug=False,
):
    class PhiEncoder(TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X: pd.DataFrame, y=None):
            # check if there is categorical feature
            for col in X.columns:
                if pd.api.types.is_categorical_dtype(X[col]):
                    break
            else:
                raise ValueError("There is no categorical feature.")
            return self

        def transform(self, X: pd.DataFrame):
            X = X.copy()
            for col in X.columns:
                if pd.api.types.is_categorical_dtype(X[col]):
                    X[col] = X[col].astype(float)
            return X

    def train_and_test(
        n_sample,
        cardinality,
        noise_std,
        x2_max,
        seed,
        model,
    ):
        df_train, y_col = gen_reg_with_cat(
            n_sample=n_sample,
            cardinality=cardinality,
            noise_std=noise_std,
            x2_max=x2_max,
            seed=seed,
        )

        df_test, _ = gen_reg_with_cat(
            n_sample=1000,
            cardinality=cardinality,
            noise_std=noise_std,
            x2_max=x2_max,
            seed=seed + 1,
        )
        # df_test = pd.DataFrame({"x1": np.arange(cardinality)})
        # df_test["y"] = df_test["x1"]
        # df_test["x1"] = df_test["x1"].astype("str").astype("category")
        # X_test = df_test[["x1"]]
        # y_test = df_test[y_col]

        cols_feature = df_train.columns.drop([y_col, "noise"], errors="ignore")
        X_train = df_train[cols_feature]
        X_test = df_test[cols_feature]
        y_train = df_train[y_col]
        y_test = df_test[y_col]

        ecd_phi = PhiEncoder().fit(X_train)
        ecd_onehot = OneHotEncoder().fit(X_train)

        X_train_phi = ecd_phi.transform(X_train)
        X_train_onehot = ecd_onehot.transform(X_train)
        X_test_phi = ecd_phi.transform(X_test)
        X_test_onehot = ecd_onehot.transform(X_test)

        ss_phi = StandardScaler().fit(X_train_phi)
        ss_onehot = StandardScaler().fit(X_train_onehot)

        X_train_phi = ss_phi.transform(X_train_phi)
        X_train_onehot = ss_onehot.transform(X_train_onehot)
        X_test_phi = ss_phi.transform(X_test_phi)
        X_test_onehot = ss_onehot.transform(X_test_onehot)
        if model == "LinearRegression":
            model_phi = LinearRegression()
            model_onehot = LinearRegression()

            model_phi.fit(X_train_phi, y_train)
            model_onehot.fit(X_train_onehot, y_train)

            # compare the parameters
            coef_phi = model_phi.coef_  # (n_feature,)
            coef_onehot = model_onehot.coef_
            intercept_phi = model_phi.intercept_
            intercept_onehot = model_onehot.intercept_

        elif model == "MLPRegressor":
            model_phi = MLPRegressor(random_state=seed)
            model_onehot = MLPRegressor(random_state=seed)

            model_phi.fit(X_train_phi, y_train)
            model_onehot.fit(X_train_onehot, y_train)

            # coefs_ is a list of length n_layers - 1. [coef_1, coef_2, ...]
            coef_phi = model_phi.coefs_[0]  # (n_feature, n_hidden_units)
            coef_onehot = model_onehot.coefs_[0]

            # intercepts_ is a list of length n_layers - 1. [intercept_1, intercept_2, ...]
            intercept_phi = model_phi.intercepts_[0]  # (n_hidden_units,)
            intercept_onehot = model_onehot.intercepts_[0]
        else:
            raise NotImplementedError

        # compare the performance on train set and test set
        y_pred_train_phi = model_phi.predict(X_train_phi)
        y_pred_train_onehot = model_onehot.predict(X_train_onehot)
        y_pred_test_phi = model_phi.predict(X_test_phi)
        y_pred_test_onehot = model_onehot.predict(X_test_onehot)

        mse_train_phi = mean_squared_error(y_train, y_pred_train_phi)
        mse_train_onehot = mean_squared_error(y_train, y_pred_train_onehot)
        mse_test_phi = mean_squared_error(y_test, y_pred_test_phi)
        mse_test_onehot = mean_squared_error(y_test, y_pred_test_onehot)

        # compare the parameters and z
        num_of_raw_numeric_feature = 0
        for col in X_train.columns:
            if not pd.api.types.is_categorical_dtype(X_train[col]):
                num_of_raw_numeric_feature += 1
        if num_of_raw_numeric_feature != 0:
            X_test_phi = X_test_phi[:, :-num_of_raw_numeric_feature]
            X_test_onehot = X_test_onehot[:, :-num_of_raw_numeric_feature]
            coef_phi = coef_phi[:-num_of_raw_numeric_feature]
            coef_onehot = coef_onehot[:-num_of_raw_numeric_feature]

        # TODO intercept include x1,x2 not just x1.
        # So the intercept is not comparable when there are x2.
        z_phi_nointercept = X_test_phi @ coef_phi
        z_phi = z_phi_nointercept + intercept_phi
        z_onehot_nointercept = X_test_onehot @ coef_onehot
        z_onehot = z_onehot_nointercept + intercept_onehot

        if debug:
            norm_z_nointercept = np.linalg.norm(
                (z_phi_nointercept - z_onehot_nointercept).reshape(-1, 1), ord=2, axis=1
            )  # (n_sample,)
            l2loss_z_nointercept = (
                norm_z_nointercept**2
            ).sum() / norm_z_nointercept.shape[0]
            norm_z = np.linalg.norm(
                (z_phi - z_onehot).reshape(-1, 1), ord=2, axis=1
            )  # (n_sample,)
            l2loss_z = (norm_z**2).sum() / norm_z.shape[0]
            assert np.isclose(
                mean_squared_error(z_phi, z_onehot), l2loss_z
            ) and np.isclose(
                mean_squared_error(z_phi_nointercept, z_onehot_nointercept),
                l2loss_z_nointercept,
            )
        else:
            l2loss_z_nointercept = mean_squared_error(
                z_phi_nointercept, z_onehot_nointercept
            )
            l2loss_z = mean_squared_error(z_phi, z_onehot)
        return (
            n_sample,
            cardinality,
            noise_std,
            x2_max,
            seed,
            model,
            l2loss_z_nointercept,
            l2loss_z,
            mse_train_phi,
            mse_train_onehot,
            mse_test_phi,
            mse_test_onehot,
        )

    if debug:
        # debug
        records = []
        for x in itertools.product(
            n_samples, cardinalitys, noise_stds, x2_maxs, seeds, models
        ):
            records.append(train_and_test(*x))
    else:
        records = Parallel(n_jobs=-1)(
            delayed(train_and_test)(*x)
            for x in itertools.product(
                n_samples, cardinalitys, noise_stds, x2_maxs, seeds, models
            )
        )

    df_res = pd.DataFrame(
        records,
        columns=[
            "n_sample",
            "cardinality",
            "noise_std",
            "x2_max",
            "seed",
            "model",
            "l2loss_z_nointercept",
            "l2loss_z",
            "mse_train_phi",
            "mse_train_onehot",
            "mse_test_phi",
            "mse_test_onehot",
        ],
    )
    df_res["sample_per_level"] = df_res["n_sample"] / df_res["cardinality"]

    return df_res


def draw_contr_diff_betw_ecd(
    df_model: pd.DataFrame,
    save_path=None,
    figsize=(4, 4),
    errorbar=("ci", 95),
    fontsize=18,
    xtick_nbins=5,
):
    """Draw contribution difference between two encoders.

    errorbar: None | ("ci" | "sd" | "sem" | "std" | "boot" | "bc" | "bca" | "percentile" | "pi", 95)
    """
    df_model["ASPL"] = df_model["n_sample"] / df_model["cardinality"]
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_model,
        x="ASPL",
        y="l2loss_z_nointercept",
        ax=ax,
        errorbar=errorbar,
    )
    sns.scatterplot(
        data=df_model.groupby("ASPL", as_index=False)["l2loss_z_nointercept"].mean(),
        x="ASPL",
        y="l2loss_z_nointercept",
        ax=ax,
    )
    ax.set_ylabel("Contribution difference")

    # "l2loss_z", "l2loss_z_nointercept" trends are the same
    # df_melt1 = df_model.melt(
    #     id_vars=["ASPL", "seed"],
    #     value_vars=["l2loss_z", "l2loss_z_nointercept"],
    #     var_name="Input at the first Affine transformation",
    #     value_name="Difference between encoders",
    # )
    # sns.lineplot(
    #     data=df_melt1,
    #     x="ASPL",
    #     y="Difference between encoders",
    #     hue="Input at the first Affine transformation",
    #     style="Input at the first Affine transformation",
    #     markers=True,
    #     dashes=False,
    #     ax=ax,
    #     errorbar=errorbar,
    # )
    if xtick_nbins is not None:
        ax.locator_params(axis="x", nbins=xtick_nbins)

    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams["font.size"] = 10


def draw_performance_diff_betw_ecd(
    df_model: pd.DataFrame,
    save_path=None,
    figsize=(4, 4),
    errorbar=None,
    fontsize=18,
    xtick_nbins=5,
):
    """Draw performance difference between two encoders.
    errorbar: None | ("ci" | "sd" | "sem" | "std" | "boot" | "bc" | "bca" | "percentile" | "pi", 95)
    """
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    df_model["Best"] = df_model["mse_test_phi"]
    df_model["OneHot"] = df_model["mse_test_onehot"]
    df_model["Performance difference"] = (
        df_model["mse_test_onehot"] - df_model["mse_test_phi"]
    )
    df_model_melt = df_model.melt(
        id_vars=["ASPL", "seed"],
        value_vars=[
            "Best",
            "OneHot",
        ],
        var_name="Encoder",
        value_name="Mean Squared Error",
    )
    fig, ax1 = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_model_melt,
        x="ASPL",
        y="Mean Squared Error",
        hue="Encoder",
        style="Encoder",
        markers=True,
        dashes=False,
        ax=ax1,
        errorbar=errorbar,
    )

    # draw two y-axis https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html

    # it may make figure ugly and chaotic
    # lineplot is not appropriate since it confuses the meanings of ax2 and ax1
    ax2 = ax1.twinx()
    tmp = df_model.groupby("ASPL")["Performance difference"].mean()
    color = "green"
    ax2.bar(tmp.index, tmp.values, alpha=0.4, color=color, width=1.5)
    ax2.set_ylabel("Performance difference", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    if xtick_nbins is not None:
        ax1.locator_params(axis="x", nbins=xtick_nbins)
        ax2.locator_params(axis="x", nbins=xtick_nbins)

    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
    if fontsize is not None:
        plt.rcParams["font.size"] = 10


def analyze_reg_with_cat_comp_onehot_phi(
    df_res: pd.DataFrame,
    save_dir=None,
):
    for model in df_res["model"].unique():
        df_model = df_res[df_res["model"] == model].copy()
        df_model["ASPL"] = df_model["n_sample"] / df_model["cardinality"]

        # ------------------- compare first Affine transformation-------------------
        save_path = (
            os.path.join(save_dir, f"contr_diff_betw_ecd_{model}.pdf")
            if save_dir is not None
            else None
        )
        draw_contr_diff_betw_ecd(
            df_model,
            save_path=save_path,
        )

        # ------------------- compare final model performance -------------------
        save_path = (
            os.path.join(save_dir, f"perf_diff_betw_ecd_{model}.pdf")
            if save_dir is not None
            else None
        )
        draw_performance_diff_betw_ecd(
            df_model,
            save_path=save_path,
            errorbar=("ci", 95),
        )


def gen_binary_cross_period(
    n_sample: int = 10000,
    x1_range: tuple = (-2, 2),
    x2_range: tuple = (-2, 2),
    flip_ratio: float = None,
    seed: int = None,
    x1_is_relavant=True,
    make_1313=False,
    use_grid=False,
    grid_step: float = None,
):
    r"""x1 is categorical, x2 is numerical.
    .. math::
        y=\operatorname{sgn}(\sin(x_1  \pi)) \cdot \operatorname{sgn}(\sin(x_2 \pi))

    """
    if not use_grid:
        if grid_step is not None:
            warnings.warn(
                "grid_step is not None, but use_grid is False. grid_step will be ignored."
            )
        seed = np.random.get_state()[1][0] if seed is None else seed
        rng = np.random.default_rng(seed)

        # cardinality = int(x1_range[1] - x1_range[0])
        # sample_per_level = int(n_sample / cardinality)
        x1 = rng.integers(x1_range[0], x1_range[1], size=n_sample) + 0.5
        x2 = rng.uniform(x2_range[0], x2_range[1], size=n_sample)

        X_train = pd.DataFrame({"x1": x1, "x2": x2})
        X_train["x1"] = X_train["x1"].astype("str").astype("category")
        if x1_is_relavant:
            y_train = np.sign(np.sin(x1 * np.pi)) * np.sign(
                np.sin(x2 * np.pi)
            )  # -1 0 1
        else:
            y_train = np.sign(np.sin(x2 * np.pi))
        if make_1313:
            assert -x2_range[0] == x2_range[1]
            index_filp = np.logical_and(x2 < 0, np.sin(x2 * np.pi) < 0)
            y_train[index_filp] = -y_train[index_filp]

        y_train = np.where(y_train == 0, -1, y_train)  # change y=0 to y=-1

        # can make figure more clear, but not consistent with the paper
        # idx_filt = y_train != 0
        # X_train = X_train[idx_filt]
        # y_train = y_train[idx_filt]

        # flip y
        if flip_ratio is not None and flip_ratio > 0:
            flip_idx = rng.choice(
                n_sample, size=int(n_sample * flip_ratio), replace=False
            )
            y_train[flip_idx] = -y_train[flip_idx]
        y_train = pd.Series(y_train, name="y").astype("int")
        return X_train, y_train

    if n_sample is not None:
        warnings.warn(
            "n_sample is not None, but use_grid is True. n_sample will be ignored."
        )
    x1_grid = np.arange(x1_range[0], x1_range[1], 1) + 0.5
    x2_grid = np.arange(x2_range[0], x2_range[1], grid_step)
    x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
    x1 = x1_grid.flatten()
    x2 = x2_grid.flatten()

    X_test = pd.DataFrame({"x1": x1, "x2": x2})
    X_test["x1"] = X_test["x1"].astype("str").astype("category")
    if x1_is_relavant:
        y_test = np.sign(np.sin(x1 * np.pi)) * np.sign(np.sin(x2 * np.pi))
    else:
        y_test = np.sign(np.sin(x2 * np.pi))
    if make_1313:
        index_filp = np.logical_and(x2 < 0, np.sin(x2 * np.pi) < 0)
        y_test[index_filp] = -y_test[index_filp]

    y_test = np.where(y_test == 0, -1, y_test)  # change y=0 to y=-1

    # can make figure more clear, but not consistent with the paper
    # idx_filt = y_test != 0
    # X_test = X_test[idx_filt]
    # y_test = y_test[idx_filt]

    y_test = pd.Series(y_test, name="y").astype("int")
    return X_test, y_test


def gen_binary_cross_period_trainandtest(
    n_sample=10000,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    flip_ratio=None,
    seed=None,
    x1_is_relavant=True,
    make_1313=False,
    test_grid_step: float = None,
):
    r"""x1 is categorical, x2 is numerical.
    .. math::
        y=\operatorname{sgn}(\sin(x_1  \pi)) \cdot \operatorname{sgn}(\sin(x_2 \pi))

    """
    X_train, y_train = gen_binary_cross_period(
        n_sample=n_sample,
        x1_range=x1_range,
        x2_range=x2_range,
        flip_ratio=flip_ratio,
        seed=seed,
        x1_is_relavant=x1_is_relavant,
        make_1313=make_1313,
        use_grid=False,
        grid_step=None,
    )

    if test_grid_step is None:
        use_grid = False
        test_n_sample = 1000
    else:
        use_grid = True
        test_n_sample = None

    X_test, y_test = gen_binary_cross_period(
        n_sample=test_n_sample,
        x1_range=x1_range,
        x2_range=x2_range,
        flip_ratio=None,
        seed=seed,
        x1_is_relavant=x1_is_relavant,
        make_1313=make_1313,
        use_grid=use_grid,
        grid_step=test_grid_step,
    )

    return X_train, y_train, X_test, y_test


def draw_binary_cross_period_groundtrue(
    x1_range=(-2, 2), x2_range=(-2, 2), x1_is_relavant=True, save_path=None, alpha=1
):
    r"""x1 is a categorical feature, and x2 is a numerical feature.
    .. math::
        y=\operatorname{sgn}(\sin(\phi(x_1) \pi)) \cdot \operatorname{sgn}(\sin(x_2 \pi))
    \phi(x_1) \in {-1.5, -0.5, 0.5, 1.5}
    """
    x1 = np.arange(x1_range[0], x1_range[1], 1)
    x2 = np.arange(x2_range[0], x2_range[1], 1)
    points = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    df = pd.DataFrame(points, columns=["x1", "x2"])

    if x1_is_relavant:
        # if x1 is in odd (even) cel, x2 is odd (even) cell, y=1
        # else -1
        df["y"] = (~np.logical_xor((df["x1"] + 1) % 2, (df["x2"] + 1) % 2)) * 2 - 1
    else:
        # if x2 is in odd cell, y=1
        # else -1
        df["y"] = ((df["x2"] + 1) % 2) * 2 - 1

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    lines = []
    cmap = ListedColormap(["#0000FF", "#FF0000"])  # Blue and Red
    # cmap = ListedColormap(["#000000", "#D3D3D3"])  # Black and LightGray
    has_add_pos_label = False
    has_add_neg_label = False
    for row in df.itertuples(index=False):
        # start_point = (row.x1+0.5, row.x2)
        # end_point = (row.x1+0.5, row.x2 + 1)
        xs = [row.x1 + 0.5, row.x1 + 0.5]
        ys = [row.x2, row.x2 + 1]
        is_pos = row.y == 1
        color = cmap(is_pos)
        label = None
        if is_pos:
            if not has_add_pos_label:
                label = "1"
                has_add_pos_label = True
        else:
            if not has_add_neg_label:
                label = "-1"
                has_add_neg_label = True
        lines.append(
            ax.add_line(
                Line2D(xs, ys, linewidth=3, color=color, alpha=alpha, label=label)
            )
        )

    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        ax.set_title("Ground truth")
        plt.show()


def draw_binary_cross_period_scatter(
    data_X: pd.DataFrame | np.ndarray,
    data_y: pd.Series | np.ndarray,
    xlim=None,
    ylim=None,
    title=None,
    save_path=None,
    figsize=None,
    fontsize=None,
    dict_season_encoding: dict = None,
    equal_space=False,
    sortby_encoding=False,
    xticklabel="num+cat",
):
    """xticklabel="num+cat" | "num" | "cat" """
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True)
    ax.set_axisbelow(True)
    if not isinstance(data_X, pd.DataFrame):
        data_X = pd.DataFrame(data_X, columns=["x1", "x2"])
    if not isinstance(data_y, pd.Series):
        data_y = pd.Series(data_y, name="y")
    if dict_season_encoding is None:
        # TODO can use int to represent season
        dict_season_encoding = {
            "spring": -1.5,
            "summer": -0.5,
            "autumn": 0.5,
            "winter": 1.5,
        }

    tmpdata = data_X.copy()
    tmpdata["x1"] = tmpdata["x1"].astype("float")
    tmpdata["y"] = data_y
    if xlim is not None:
        tmpdata = tmpdata[tmpdata["x1"].between(*xlim)]
    if ylim is not None:
        tmpdata = tmpdata[tmpdata["x2"].between(*ylim)]
    hue_order = np.unique(tmpdata["y"])
    if equal_space:
        ser_season_encoding = pd.Series(dict_season_encoding)
        if sortby_encoding:
            ser_season_encoding.sort_values(ascending=True, inplace=True)
            ser_season_position = ser_season_encoding.rank(method="dense").astype(int)
        else:
            # TODO d_season_ecd={"spring":1, "summer":-1, "autumn":1, "winter":-1"} the figure is wrong
            ser_season_position = pd.Series(
                range(1, len(ser_season_encoding) + 1), index=ser_season_encoding.index
            )

        # encoding -> position
        tmpdict = {}
        for season, encoding in ser_season_encoding.items():
            tmpdict[encoding] = ser_season_position.loc[season]
        tmpdata["x1"] = tmpdata["x1"].map(tmpdict)
    else:
        ser_season_position = pd.Series(dict_season_encoding).sort_values(
            ascending=True
        )

    sns.scatterplot(
        x="x1",
        y="x2",
        hue="y",
        data=tmpdata,
        ax=ax,
        palette=ListedColormap(["#0000FF", "#FF0000"]),
        hue_order=hue_order,
        style="y",
        markers=["o", "X"],
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xticklabel == "num":
        # {"spring":0.5, "summer":0.5, "fall":1} -> {0.5 : 0.5, 1:1}
        pass
    elif xticklabel == "cat":
        # {"spring":0.5, "summer":0.5, "fall":1} -> {0.5 : "spring\nsummer", 1:"fall"}
        dict_pos_label = {}
        for season, pos in ser_season_position.items():
            dict_pos_label[pos] = dict_pos_label.get(pos, "") + season  # TODO \n
    elif xticklabel == "num+cat":
        dict_pos_label = {}
        # {"spring":0.5, "summer":0.5, "fall":1} -> {0.5 : "0.5\nspring\nsummer", 1: "1\nfall"}
        for season, pos in ser_season_position.items():
            if pos not in dict_pos_label:
                encoding = dict_season_encoding[season]
                dict_pos_label[pos] = f"{encoding:.4f}" + "\n" + season
            else:
                dict_pos_label[pos] = dict_pos_label[pos] + "\n" + season
    else:
        raise ValueError(f"xticklabel={xticklabel} not supported.")

    x_tick_positions = list(dict_pos_label.keys())
    x_labels = list(dict_pos_label.values())
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_labels)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams["font.size"] = 10


def draw_decision_boundary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    clf=None,
    encoder=None,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    save_path=None,
):
    assert X_train.shape[1] == 2
    if clf is None:
        clf = RandomForestClassifier(random_state=42)
    if encoder is None:
        encoder = MeanEncoder()
    y_train_forecd = utils.format_y_binary(
        y_train
    )  # [-1,1] need to convert to [0,1] for TargetEncoder
    X_train_ecded = encoder.fit_transform(X_train, y_train_forecd)
    clf.fit(X_train_ecded, y_train_forecd)

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    X_test_ended = encoder.transform(X_test)
    y_test_pred = clf.predict(X_test_ended)

    xx0 = X_test["x1"].astype(float).to_numpy().reshape(-1, x1_range[1] - x1_range[0])
    xx1 = X_test["x2"].to_numpy().reshape(-1, x1_range[1] - x1_range[0])
    display = DecisionBoundaryDisplay(
        xx0=xx0, xx1=xx1, response=y_test_pred.reshape(xx0.shape)
    )
    # display.plot(ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True), alpha=0.5)
    display.plot(
        ax=ax,
        plot_method="contour",  # 'contourf', 'contour', 'pcolormesh'
        #  cmap=sns.color_palette("coolwarm", as_cmap=True)
        # cmap=ListedColormap(["#000000", "#D3D3D3"]),  # Black and LightGray
        cmap=ListedColormap(["#0000FF", "#FF0000"]),
    )

    # only work for X is 2D numerical
    # tmpx1 = np.arange(x1_range[0], x1_range[1] + 1, 1)
    # tmpx2 = np.arange(x2_range[0], x2_range[1] + 1, 1)
    # tmpX = np.array(np.meshgrid(tmpx1, tmpx2)).T.reshape(-1, 2)
    # display = DecisionBoundaryDisplay.from_estimator(
    #     clf,
    #     pd.DataFrame(tmpX, columns=["x1", "x2"]),
    #     # fill_alpha=0.5,
    #     cmap=ListedColormap(["#0000FF", "#FF0000"]),
    #     ax=ax,
    #     plot_method="contour",  # 'contourf', 'contour', 'pcolormesh'
    #     response_method="predict",  # 'auto', 'predict_proba', 'decision_function', 'predict'
    # )

    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)

    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()


def draw_binary_cross_period_overfit(
    debug=False, figsize=None, fontsize=None, equal_space=False, xticklabel="num+cat"
):
    x1_range = (-2, 2)
    # x2_range = (-2, 2)
    # n_samples = (12, 40, 120)
    # seed = 0
    x2_range = (-2, 3)
    n_samples = (21, 40, 120, 200)
    seed = 0
    output_dir = "../output/paper/assets"
    for x1_is_relavant in (True,):
        for n_sample in n_samples:
            sample_per_level = int(n_sample / (x1_range[1] - x1_range[0]))
            prefix = "relavant" if x1_is_relavant else "irrelavant"

            X_train, y_train, X_test, y_test = gen_binary_cross_period_trainandtest(
                n_sample=n_sample,
                x1_range=x1_range,
                x2_range=x2_range,
                x1_is_relavant=x1_is_relavant,
                flip_ratio=None,
                seed=seed,
            )
            ecd = MeanEncoder()
            clf = RandomForestClassifier(random_state=seed)
            ecd.fit(X_train, utils.format_y_binary(y_train, neg_and_pos=False))
            X_train_ecded = ecd.transform(X_train)
            X_test_ecded = ecd.transform(X_test)
            clf.fit(X_train_ecded, y_train)

            y_train_pred = clf.predict(X_train_ecded)
            y_test_pred = clf.predict(X_test_ecded)

            save_path_groundtrue = os.path.join(
                output_dir, f"{prefix}_cross_period_groundtruth.png"
            )
            save_path_train = os.path.join(
                output_dir,
                f"{prefix}_cross_period_train_{sample_per_level}.png",
            )
            save_path_testpred = os.path.join(
                output_dir,
                f"{prefix}_cross_period_testpred_{sample_per_level}.png",
            )
            msg = prefix
            msg += f" ASPL:{sample_per_level}"
            msg += f" TrainAcc: {accuracy_score(y_train, y_train_pred):.4f}"
            msg += f" TestAcc: {accuracy_score(y_test, y_test_pred):.4f}"
            if debug:
                save_path_groundtrue = None
                save_path_train = None
                save_path_testpred = None
                title = msg
            else:
                title = None
                print(msg)

            # TODO only need to draw once
            # draw_binary_cross_period_groundtrue(x2_range=x2_range,x1_is_relavant=x1_is_relavant)
            draw_binary_cross_period_scatter(
                X_test,
                y_test,
                ylim=x2_range,
                save_path=save_path_groundtrue,
                title=title,
                figsize=figsize,
                fontsize=fontsize,
                equal_space=equal_space,
                xticklabel=xticklabel,
            )
            # if debug:
            draw_binary_cross_period_scatter(
                X_train,
                y_train,
                ylim=x2_range,
                title=title,
                figsize=figsize,
                fontsize=fontsize,
                equal_space=equal_space,
                xticklabel=xticklabel,
                save_path=save_path_train,
            )
            draw_binary_cross_period_scatter(
                X_test,
                y_test_pred,
                ylim=x2_range,
                save_path=save_path_testpred,
                title=title,
                figsize=figsize,
                fontsize=fontsize,
                equal_space=equal_space,
                xticklabel=xticklabel,
            )
            if debug:
                draw_decision_boundary(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    clf=clf,
                    encoder=ecd,
                    x1_range=x1_range,
                    x2_range=x2_range,
                )


def exp_binary_cross_period_diff_aspl_and_ecd(
    n_samples=range(20, 201, 4),
    x1_is_relavants=(True, False),
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    make_1313s=(False,),
    flip_ratios=(0,),
    models=(
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
    ),
    seeds=range(10),
    encoders=(
        "BestEncoder",
        "DeleteEncoder",
        "MeanEncoder",
        "TargetEncoder",
        "OneHotEncoder",
    ),
    save_path="../output/drawback_of_te/binary_cross_period_diff_aspl_and_ecd.csv",
):
    assert (
        x1_range[0] < x1_range[1]
        and isinstance(x1_range[0], int)
        and isinstance(x1_range[1], int)
    )
    assert (
        x2_range[0] < x2_range[1]
        and isinstance(x2_range[0], int)
        and isinstance(x2_range[1], int)
    )
    assert all(
        isinstance(flip_ratio, int) and 0 <= flip_ratio <= 100
        for flip_ratio in flip_ratios
    )

    def train_and_test(
        n_sample, x1_is_relavant, make_1313, flip_ratio, seed, encoder, model
    ):
        X_train, y_train, X_test, y_test = gen_binary_cross_period_trainandtest(
            n_sample=n_sample,
            x1_range=x1_range,
            x2_range=x2_range,
            seed=seed,
            x1_is_relavant=x1_is_relavant,
            make_1313=make_1313,
            flip_ratio=flip_ratio / 100,
        )

        if model == "DecisionTreeClassifier":
            clf = DecisionTreeClassifier(random_state=seed)
        elif model == "RandomForestClassifier":
            clf = RandomForestClassifier(random_state=seed)
        elif model == "LGBMClassifier":
            clf = LGBMClassifier(random_state=seed)
        elif model == "XGBClassifier":
            clf = XGBClassifier(random_state=seed)
        else:
            raise ValueError(f"model {model} not supported.")

        if encoder == "BestEncoder":

            class BestEncoder(TransformerMixin):
                """{..., "-1.5", "-0.5", "0.5", "1.5",...} -> {..., 1, -1, 1, -1, ...} or
                {..., (0, 1), (1, 0) , (0, 1), (1, 0),...}
                """

                def __init__(self):
                    pass

                def fit(self, X: pd.DataFrame, y=None):
                    return self

                def transform(self, X: pd.DataFrame):
                    X = X.copy()

                    # def _f(x: float):
                    #     return pd.Series(
                    #         ((x - 0.5) % 2, (x + 0.5) % 2), index=["x1_0", "x1_1"]
                    #     )

                    # X[["x1_0", "x1_1"]] = X["x1"].astype("float").apply(_f)
                    # X.drop(columns=["x1"], inplace=True)
                    X["x1"] = ((X["x1"].astype(float) + 0.5) % 2) * 2 - 1
                    return X

            ecd = BestEncoder()
        elif encoder == "DeleteEncoder":
            ecd = DeleteEncoder(delete_type="highest_card")
        elif encoder == "MeanEncoder":
            ecd = MeanEncoder()
        else:
            ecd = getattr(ce, encoder)()

        y_train = utils.format_y_binary(y_train, neg_and_pos=False)
        y_test = utils.format_y_binary(y_test, neg_and_pos=False)
        X_train_ecded = ecd.fit_transform(X_train, y_train)
        X_test_ecded = ecd.transform(X_test)

        clf.fit(X_train_ecded, y_train)
        y_train_pred = clf.predict(X_train_ecded)
        y_test_pred = clf.predict(X_test_ecded)

        acc = accuracy_score(y_test, y_test_pred)
        acc_train = accuracy_score(y_train, y_train_pred)
        f1 = f1_score(y_test, y_test_pred)
        f1_train = f1_score(y_train, y_train_pred)

        cardinality = x1_range[1] - x1_range[0]
        rangeofx2 = x2_range[1] - x2_range[0]
        exp_id = "-".join(
            map(
                str,
                (
                    cardinality,
                    rangeofx2,
                    n_sample,
                    x1_is_relavant,
                    make_1313,
                    flip_ratio,
                    seed,
                    encoder,
                    model,
                ),
            )
        )
        return (
            exp_id,
            cardinality,
            rangeofx2,
            n_sample,
            x1_is_relavant,
            make_1313,
            flip_ratio,
            seed,
            model,
            encoder,
            acc,
            acc_train,
            f1,
            f1_train,
        )

    columns = [
        "exp_id",
        "cardinality",
        "rangeofx2",
        "n_sample",
        "x1_is_relavant",
        "make_1313",
        "flip_ratio",
        "seed",
        "model",
        "encoder",
        "acc",
        "acc_train",
        "f1",
        "f1_train",
    ]

    cardinality = x1_range[1] - x1_range[0]
    rangeofx2 = x2_range[1] - x2_range[0]
    params = list(
        itertools.product(
            (cardinality,),
            (rangeofx2,),
            n_samples,
            x1_is_relavants,
            make_1313s,
            flip_ratios,
            seeds,
            encoders,
            models,
        )
    )
    exp_ids = ["-".join(map(str, param)) for param in params]
    if os.path.exists(save_path):
        df_res = pd.read_csv(save_path)
        exp_ids_done = df_res["exp_id"].unique()
        params_idxs = [
            i for i, exp_id in enumerate(exp_ids) if exp_id not in exp_ids_done
        ]
        params2exp = [params[i] for i in params_idxs]
    else:
        df_res = pd.DataFrame(columns=columns)
        params2exp = params

    records = Parallel(n_jobs=-1)(
        delayed(train_and_test)(
            n_sample, x1_is_relavant, make_1313, flip_ratio, seed, encoder, model
        )
        for cardinality, rangeofx2, n_sample, x1_is_relavant, make_1313, flip_ratio, seed, encoder, model in params2exp
    )

    df_res = pd.concat([df_res, pd.DataFrame(records, columns=columns)])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_res.to_csv(save_path, index=False)
    return df_res


def filt_dfres(
    df_res: pd.DataFrame,
    n_samples=range(20, 401, 20),
    x1_range=(-2, 2),
    x2_range=(-2, 3),
    make_1313=False,
    flip_ratio=0,
    models=("RandomForestClassifier",),
    seeds=range(30),
    encoders=(
        "BestEncoder",
        "TargetEncoder",
        "MeanEncoder",
    ),
    encoders2compare=("BestEncoder", "MeanEncoder"),
    rename_targetencoder=True,
):
    df_res = df_res[
        df_res["n_sample"].isin(n_samples)
        & df_res["seed"].isin(seeds)
        & df_res["encoder"].isin(encoders)
        & df_res["model"].isin(models)
        & (df_res["cardinality"] == (x1_range[1] - x1_range[0]))
        & (df_res["rangeofx2"] == (x2_range[1] - x2_range[0]))
        & (df_res["make_1313"] == make_1313)
        & (df_res["flip_ratio"] == flip_ratio)
    ].copy()
    df_res["ASPL"] = df_res["n_sample"] / df_res["cardinality"]

    df_res["encoder"] = df_res["encoder"].str.replace("Encoder", "")
    encoders = [encoder.replace("Encoder", "") for encoder in encoders]  # for plot
    encoders2compare = (
        [encoder.replace("Encoder", "") for encoder in encoders2compare]
        if encoders2compare is not None
        else None
    )
    df_res["model"] = df_res["model"].str.replace("Classifier", "")

    if rename_targetencoder:
        df_res["encoder"] = df_res["encoder"].str.replace("Target", "SShrink")
        encoders = [encoder.replace("Target", "SShrink") for encoder in encoders]
        encoders2compare = (
            [encoder.replace("Target", "SShrink") for encoder in encoders2compare]
            if encoders2compare is not None
            else None
        )
    return df_res, encoders, encoders2compare


def draw_binary_cross_period_diff_aspl_and_ecd(
    df_res: pd.DataFrame,
    # save_path="../output/drawback_of_te/binary_cross_period_diff_aspl_and_ecd.csv",
    output_dir="../output/drawback_of_te/",
    x1_is_relavants=(True,),
    encoders=(
        "Best",
        "SShrink",
        "Mean",
    ),
    encoders2compare=("Best", "Mean"),  # ("SShrink", "Mean") looks weird
    fontsize=20,
):
    df_res = df_res[df_res["encoder"].isin(encoders)].copy()
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize

    for x1_is_relavant in x1_is_relavants:
        for model in df_res["model"].unique():
            df_tmp = df_res[df_res["model"] == model]
            df_tmp = df_tmp[df_tmp["x1_is_relavant"] == x1_is_relavant]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(
                data=df_tmp,
                x="ASPL",
                y="acc",
                hue="encoder",
                hue_order=encoders,
                style="encoder",
                markers=True,
                dashes=False,
                ax=ax,
            )
            # ax.set_xlabel("ASPL")
            ax.set_xlabel("ASPL")
            ax.set_ylabel("Accuracy")
            # ax.get_legend().set_title("Encoder")
            if encoders2compare is not None:
                ax.legend(loc="center")
                ax1 = ax.twinx()
                df_tmp = df_tmp[df_tmp["encoder"].isin(encoders2compare)]

                def f(df: pd.DataFrame):
                    df_2record = df.set_index("encoder")
                    return (
                        df_2record.loc[encoders2compare[0], "acc"]
                        - df_2record.loc[encoders2compare[1], "acc"]
                    )

                df_tmp = (
                    df_tmp.groupby(["ASPL", "seed"])
                    .apply(f)
                    .reset_index()
                    .rename(columns={0: "diff"})
                )

                color = "green"
                # sns.lineplot(
                #     data=df_tmp,
                #     x="ASPL",
                #     y="diff",
                #     ax=ax1,
                #     color=color,
                #     marker="*",
                #     # label="Diff",
                # )
                tmp = df_tmp.groupby("ASPL")["diff"].mean()
                ax1.bar(tmp.index, tmp.values, alpha=0.4, color=color, width=1.5)
                ax1.tick_params(axis="y", labelcolor=color)
                ax1.set_xlabel("ASPL")
                ax1.set_ylabel("Performance difference", color=color)

            ax.get_legend().set_title("Encoder")
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                prefix = "relavant" if x1_is_relavant else "irrelavant"
                for path in utils.convert_pathoffig(
                    os.path.join(output_dir, f"aspl_perf_tree_ecd_{prefix}_{model}.png")
                ):
                    plt.savefig(path, bbox_inches="tight")
                plt.close("all")
            else:
                plt.show()

    if fontsize is not None:
        plt.rcParams["font.size"] = 10

    return df_res


def draw_binary_cross_compare_ecd(
    df_res: pd.DataFrame,
    encoders2compare=("TargetEncoder", "MeanEncoder"),
    fontsize=20,
    figsize=None,
    save_path=None,
):
    df_res = df_res.copy()
    df_res["ASPL"] = df_res["n_sample"] / df_res["cardinality"]
    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize

    df_tmp = df_res[df_res["encoder"].isin(encoders2compare)]

    def f(df: pd.DataFrame):
        df_2record = df.set_index("encoder")
        return (
            df_2record.loc[encoders2compare[0], "acc"]
            - df_2record.loc[encoders2compare[1], "acc"]
        )

    df_tmp = (
        df_tmp.groupby(["ASPL", "seed"])
        .apply(f)
        .reset_index()
        .rename(columns={0: "diff"})
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(df_tmp, x="ASPL", y="diff", ax=ax)
    sns.scatterplot(
        df_tmp.groupby("ASPL", as_index=False)[["diff"]].mean(),
        x="ASPL",
        y="diff",
        ax=ax,
    )
    ax.axhline(0, color="r", linestyle="--")
    # ax.set_xlabel("ASPL")
    ax.set_xlabel("ASPL")
    ax.set_ylabel("Performance Difference")

    if save_path is not None:
        for path in utils.convert_pathoffig(save_path):
            plt.savefig(path, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams["font.size"] = 10


def draw_ord_of_diff_ecd(
    n_sample=40,
    x2_range=(-2, 3),
    seed=40,
    draw=True,
    save=False,
    fontsize=None,
    figsize=None,
    equal_space=False,
    sortby_encoding=True,
):
    X_train, y_train, X_test, y_test = gen_binary_cross_period_trainandtest(
        n_sample=n_sample,
        x1_range=(-2, 2),
        x2_range=x2_range,
        x1_is_relavant=True,
        flip_ratio=None,
        seed=seed,
    )
    y_train_forecd = utils.format_y_binary(y_train)
    df_summary = X_train.copy()

    ecd_mean = MeanEncoder()
    X_train_mean = ecd_mean.fit_transform(X_train, y_train_forecd)
    X_test_mean = ecd_mean.transform(X_test)
    df_summary["x1_ecded_mean"] = X_train_mean["x1"]

    ecd_target = TargetEncoder()
    X_train_target = ecd_target.fit_transform(X_train, y_train_forecd)
    X_test_target = ecd_target.transform(X_test)
    df_summary["x1_ecded_target"] = X_train_target["x1"]

    class BestEncoder(TransformerMixin):
        """{..., "-1.5", "-0.5", "0.5", "1.5",...} -> {..., 1, -1, 1, -1, ...} or
        {..., a, b , a, b,...}
        """

        def __init__(self, dict_for_map=None):
            self.dict_for_map = dict_for_map

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X = X.copy()
            X["x1"] = ((X["x1"].astype(float) + 0.5) % 2) * 2 - 1
            if self.dict_for_map is not None:
                X["x1"] = X["x1"].map(self.dict_for_map)

            return X

    # ecd_best = BestEncoder(dict_for_map={1: 0.6, -1: 0.4})
    ecd_best = BestEncoder(dict_for_map={1: 1, -1: -1})
    X_train_best = ecd_best.fit_transform(X_train, y_train_forecd)
    X_test_best = ecd_best.transform(X_test)
    df_summary["x1_ecded_best"] = X_train_best["x1"]

    df_summary = (
        df_summary[["x1", "x1_ecded_mean", "x1_ecded_target", "x1_ecded_best"]]
        .groupby("x1")
        .first()
    )
    for col in df_summary:
        df_summary[f"{col}_rank"] = rankdata(df_summary[col])
    df_summary["season"] = df_summary.index.map(
        {
            "-1.5": "spring",
            "-0.5": "summer",
            "0.5": "autumn",
            "1.5": "winter",
        }
    )
    df_summary["best_order"] = df_summary.index.map(
        {
            "-1.5": 3,
            "-0.5": 1,
            "0.5": 3,
            "1.5": 1,
        }
    )
    df_summary["best_proba"] = df_summary.index.map(
        {
            "-1.5": 0.6,
            "-0.5": 0.4,
            "0.5": 0.6,
            "1.5": 0.4,
        }
    )
    df_summary.reset_index(inplace=True)
    df_summary.set_index("season", inplace=True)
    df_summary = df_summary.loc[
        ["spring", "summer", "autumn", "winter"], :
    ]  # sort by season
    if draw:
        for data_X, data_y, save_path, title in zip(
            (X_test, X_train),
            (y_test, y_train),
            (
                "../output/paper/assets/binary_cross_testset.pdf",
                "../output/paper/assets/binary_cross_trainset.pdf",
            ),
            ("TestSet", "TrainSet"),
        ):
            draw_binary_cross_period_scatter(
                data_X,
                data_y,
                ylim=x2_range,
                save_path=save_path if save else None,
                title=None if save else title,
                fontsize=fontsize,
                figsize=figsize,
                equal_space=equal_space,
                sortby_encoding=sortby_encoding,
                xticklabel="cat",
            )

        for data_X, data_y, save_path, title, dict_season_encoding in zip(
            (X_train_mean, X_train_target, X_train_best),
            (y_train, y_train, y_train),
            (
                "../output/paper/assets/binary_cross_mean.pdf",
                "../output/paper/assets/binary_cross_target.pdf",
                "../output/paper/assets/binary_cross_best.pdf",
            ),
            ("TrainSet Mean", "TrainSet Target", "TrainSet Best"),
            (
                df_summary["x1_ecded_mean"].to_dict(),
                df_summary["x1_ecded_target"].to_dict(),
                df_summary["x1_ecded_best"].to_dict(),
            ),
        ):
            draw_binary_cross_period_scatter(
                data_X,
                data_y,
                ylim=x2_range,
                dict_season_encoding=dict_season_encoding,
                save_path=save_path if save else None,
                title=None if save else title,
                fontsize=fontsize,
                figsize=figsize,
                equal_space=equal_space,
                sortby_encoding=sortby_encoding,
                xticklabel="num+cat",
            )

        for X_train_process, y_train, save_path, title, X_test_process in zip(
            (X_train_mean, X_train_target, X_train_best),
            (y_train, y_train, y_train),
            (
                "../output/paper/assets/binary_cross_mean_pred.pdf",
                "../output/paper/assets/binary_cross_target_pred.pdf",
                "../output/paper/assets/binary_cross_best_pred.pdf",
            ),
            ("Prediction Mean", "Prediction Target", "Prediction Best"),
            (X_test_mean, X_test_target, X_test_best),
        ):
            model = RandomForestClassifier(random_state=seed)
            model.fit(X_train_process, y_train)
            y_train_pred = model.predict(X_train_process)
            y_test_pred = model.predict(X_test_process)
            acc_train = accuracy_score(y_train, y_train_pred)
            acc_test = accuracy_score(y_test, y_test_pred)

            draw_binary_cross_period_scatter(
                X_test,
                y_test_pred,
                ylim=x2_range,
                save_path=save_path if save else None,
                title=None if save else title,
                fontsize=fontsize,
                figsize=figsize,
                equal_space=equal_space,
                sortby_encoding=sortby_encoding,
                xticklabel="cat",
            )

            print(f"{title}. TrainAcc: {acc_train}, TestAcc: {acc_test}")

    return df_summary


def output_final():
    print("`output_final` begin")
    print("`analyze_reg_with_cat_comp_onehot_phi` begin")
    df_res = exp_reg_with_cat_comp_onehot_phi(
        models=("LinearRegression", "MLPRegressor"),
        n_samples=range(20, 401, 20),
        cardinalitys=(4,),
        noise_stds=(1,),
        x2_maxs=(None,),
        seeds=range(30),
    )
    analyze_reg_with_cat_comp_onehot_phi(df_res, save_dir="../output/paper/assets")
    print("`analyze_reg_with_cat_comp_onehot_phi` done")

    print("`draw_ord_of_diff_ecd` begin")
    draw_ord_of_diff_ecd(
        n_sample=40,
        seed=21,
        draw=True,
        save=True,
        equal_space=True,
        sortby_encoding=False,
        fontsize=22,
        figsize=None,
    )
    print("`draw_ord_of_diff_ecd` done")

    print("`draw_binary_cross_period_diff_aspl_and_ecd` begin")
    n_samples = range(20, 401, 20)
    x1_is_relavants = (True,)
    x1_range = (-2, 2)
    x2_range = (-2, 3)
    make_1313 = False
    flip_ratio = 0
    models = ("RandomForestClassifier",)
    seeds = range(30)
    encoders = (
        "BestEncoder",
        "TargetEncoder",
        "MeanEncoder",
    )
    save_path = "../output/drawback_of_te/binary_cross_period_diff_aspl_and_ecd.csv"
    df_res = exp_binary_cross_period_diff_aspl_and_ecd(
        n_samples=n_samples,
        x1_is_relavants=x1_is_relavants,
        x1_range=x1_range,
        x2_range=x2_range,
        make_1313s=(make_1313,),
        flip_ratios=(flip_ratio,),
        models=models,
        seeds=seeds,
        encoders=encoders,
        save_path=save_path,
    )
    df_res_filt, _, _ = filt_dfres(
        df_res,
        n_samples=n_samples,
        x1_range=x1_range,
        x2_range=x2_range,
        make_1313=make_1313,
        flip_ratio=flip_ratio,
        models=models,
        seeds=seeds,
        encoders=encoders,
        encoders2compare=None,
        rename_targetencoder=True,
    )
    draw_binary_cross_period_diff_aspl_and_ecd(
        df_res_filt,
        output_dir="../output/paper/assets",
        # output_dir=None,
        x1_is_relavants=(True,),
        encoders=(
            "Best",
            # "SShrink",
            "Mean",
        ),
        encoders2compare=("Best", "Mean"),
    )

    draw_binary_cross_compare_ecd(
        df_res_filt,
        encoders2compare=("SShrink", "Mean"),
        fontsize=20,
        figsize=None,
        save_path="../output/paper/assets/mean_target_acc_diff.pdf",
    )
    print("`draw_binary_cross_period_diff_aspl_and_ecd` done")


if __name__ == "__main__":
    output_final()
