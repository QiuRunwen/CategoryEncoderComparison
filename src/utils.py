"""
The utils function used in src
"""

import csv
import json
import os
import pickle
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import yaml


def save(obj: object, file_path: str):
    """save object to file_path as pickle file or json file or yaml file"""

    name, extension = os.path.splitext(file_path)
    if not extension:
        extension = ".pkl"
        file_path = name + extension

    supported_extensions = [".yaml", ".json", ".pkl"]

    # check extension
    if extension not in supported_extensions:
        raise ValueError(f"extension: {extension} not in {supported_extensions}")

    # create dir
    dir_name = os.path.dirname(file_path)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    # save
    if extension == ".json":
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(obj, file)
    elif extension == ".pkl":
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    elif extension == ".yaml":
        # avoid reference cycle
        # yaml.Dumper.ignore_aliases = lambda *args : True
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(obj, file)
    else:
        raise ValueError(f"file_extension: {extension} not in {supported_extensions}")


def load(file_path: str):
    """load object from pickle file or json file"""

    supported_extensions = [".yaml", ".json", ".pkl"]

    # check file_path
    name, _ = os.path.splitext(file_path)
    if not os.path.exists(file_path):
        for supported_extension in supported_extensions:
            tmp_file_path = name + supported_extension
            if os.path.exists(tmp_file_path):
                file_path = tmp_file_path
                break

    # load object
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            obj = json.load(file)
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
    elif file_path.endswith(".yaml"):
        with open(file_path, "r", encoding="utf-8") as file:
            obj = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"file_path: {file_path} not end with {supported_extension}")

    return obj


def dict_to_json(dict_: dict):
    """convert dict to json"""
    return json.dumps(dict_)


def append_dicts_to_csv(file_path: str, data_dicts: list[dict]):
    """Append dicts to csv file. If the file is empty, write header using the keys of data_dicts.
    if the file is not empty, the first line of the file must be the header.
    and the keys of data_dicts must be the same as the header in the file.
    e.g. data_dicts=[{"a": 1, "b": 2}, {"a": 3, "b": 4}] and the header is ["a", "b"]
    """
    if not data_dicts:
        return  # Return if the list is empty

    dir_name = os.path.dirname(file_path)
    if len(dir_name) > 0:
        os.makedirs(
            os.path.dirname(file_path), exist_ok=True
        )  # Create dir if not exists
    with open(file_path, "a+", newline="", encoding="utf-8") as csv_file:
        if csv_file.tell() == 0:
            # The file is empty, write header
            fieldnames = data_dicts[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
        else:
            # The file is not empty, get the header first and then write
            csv_file.seek(0)  # Move the cursor to the start of the file
            fieldnames = csv.DictReader(csv_file).fieldnames  # Read the header
            csv_file.seek(0, 2)  # Move the cursor to the end of the file
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerows(data_dicts)


def seed_everything(seed: int):
    """seed all random function"""
    if seed is None:
        return

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # if is_torch_installed:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)


def random_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_rate: float = None,
    sample_num: int = None,
    stratify: bool = False,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_rate is None:
        sample_rate = sample_num / len(X)

    if sample_rate >= 1:
        return X, y

    y_stratify = y if stratify else None

    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=sample_rate, stratify=y_stratify, random_state=seed
    )

    return X_sub, y_sub


def format_y_binary(y: np.ndarray, neg_and_pos: bool = False) -> np.ndarray:
    """将二分类标签转换为 0, 1 or -1, 1"""

    y_unique = np.unique(y)
    assert y_unique.size == 2

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if 1 in y_unique:
        if neg_and_pos:
            y_final = np.where(y == 1, 1, -1)
        else:
            y_final = np.where(y == 1, 1, 0)
    else:
        y_final = LabelEncoder().fit_transform(y)
        if neg_and_pos:
            y_final = y_final * 2 - 1

    return y_final


def convert_pathoffig(path):
    """convert path of fig to paths that contain .png and .pdf"""
    path = Path(path)
    path_png = path.with_suffix(".png")
    path_pdf = path.with_suffix(".pdf")
    return path_png, path_pdf


def concat_all_results(
    result_dir: str,
    drop_expid_duplicated=False,
    save_path: str = None,
):
    """Concat all results in result_dir to a csv file."""
    # find all `result.csv` files
    paths = []
    for root, dirs, files in os.walk(result_dir):
        if "result.csv" in files:
            paths.append(os.path.join(root, "result.csv"))

    # read all `result.csv` files
    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    if drop_expid_duplicated:
        df = df.drop_duplicates(subset=["exp_id"])

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"save result to {os.path.abspath(save_path)}.")

        # output the log file
        log_path = os.path.join(os.path.dirname(save_path), "concat_log.txt")
        msg = f"result_dir: {result_dir}\n"
        msg += f"drop_expid_duplicated: {drop_expid_duplicated}\n"
        msg += f"save_path: {save_path}\n"
        msg += f"len(paths): {len(paths)}\n"
        msg += f"len(df): {len(df)}\n"

        # detail file path
        msg += "\n"
        msg += "detail file path:\n"
        for path in paths:
            msg += f"{path}\n"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(msg)

        print(f"save log to {os.path.abspath(log_path)}.")

    return df


def calc_length_of_each_row_in_a_file(filepath):
    """Calculate the length of each row in a file"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lengths = []
    for line in lines:
        lengths.append(len(line))
    return lengths


def preprocess_raw_result(result_dir, min_length=150, save_dir=None):
    """Preprocess the raw result in result_dir, including:
    1. copy the result_dir into a tmp dir
    2. calculate the length of each row in each csv file in the tmp dir
    3. remove the rows whose length is less than 150 in a file.
    """
    if save_dir is None:
        save_dir = result_dir + "_preprocessed"
    os.makedirs(save_dir, exist_ok=True)
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                savepath = filepath.replace(result_dir, save_dir)
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                head = lines[0]
                records = [
                    line
                    for line in lines[1:]
                    if len(line) >= min_length and (line[1] == "-")
                ]
                lines = [head] + records
                with open(savepath, "w", encoding="utf-8") as f:
                    f.writelines(lines)


if __name__ == "__main__":
    # concat_all_results(
    #     result_dir="../output/result/current/raw",
    #     drop_expid_duplicated=True,
    #     save_path="../output/result/current/result.csv",
    # )

    preprocess_raw_result("../output/result/20231031_delete_highest/raw")

    concat_all_results(
        result_dir="../output/result/20231031_delete_highest/raw_preprocessed",
        drop_expid_duplicated=True,
        save_path="../output/result/20231031_delete_highest/result.csv",
    )
