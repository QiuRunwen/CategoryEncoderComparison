import os
import pandas as pd


def concat_all_results(
    result_dir: str, drop_expid_duplicated=False, save_path: str = None
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
    return df


# df = concat_all_results("./raw/MLP_noes", True)  # MLP without early stopping

# # remove mean encoder since the implementation is not correct
# df = df[df["encoder"] != "MeanEncoder"]

# df_meanencoder = concat_all_results("./raw/20231205_mean_encoder_msidown3", True)

# df_lnr_lr = concat_all_results("./raw/20231217_lgr_lnr_msidown1", True)

# df = pd.concat([df, df_meanencoder, df_lnr_lr], ignore_index=True)

# df.drop_duplicates(subset=["exp_id"]).to_csv("./result.csv", index=False)

df = concat_all_results("./raw/20231218", True)
df = df[~df["dataset"].isin(["UkAir", "BikeSharing"])]
df_tmp = concat_all_results("./raw/20240104_ukair", True)
df = pd.concat([df, df_tmp], ignore_index=True)
df_tmp = concat_all_results("./raw/20240105_bikesharing", True)
df = pd.concat([df, df_tmp], ignore_index=True)

df[df["encoder"] != "Delete"].to_csv(
    "./result.csv", index=False
)  # remove `Delete` since we have `DeleteEncoder`
