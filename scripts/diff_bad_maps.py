import os
import pandas as pd
from tqdm import tqdm
from yaml import safe_load
from munch import munchify
from pre.visualize import draw


with open("config.yaml") as f:
    cfg = munchify(safe_load(f))
root = cfg.fits_path
s_df = pd.read_csv("csv/maps_apr25_test_s_band.csv")
s_df_upd = pd.read_csv("csv/maps_apr25_test_upd_s_band.csv")

x_df = pd.read_csv("csv/maps_apr25_test_x_band.csv")
x_df_upd = pd.read_csv("csv/maps_apr25_test_upd_x_band.csv")

s_diff = set(s_df_upd.file_name) - set(s_df.file_name)
x_diff = set(x_df_upd.file_name) - set(x_df.file_name)

s_df_filt = s_df_upd[s_df_upd["file_name"].isin(s_diff)]
x_df_filt = x_df_upd[x_df_upd["file_name"].isin(x_diff)]

s_df_filt.to_csv("./diff/s_diff.csv")
x_df_filt.to_csv("./diff/x_diff.csv")
exit(0)

s_diff_path = "./diff/s_band"
x_diff_path = "./diff/x_band"
os.makedirs(s_diff_path, exist_ok=True)
os.makedirs(x_diff_path, exist_ok=True)

for source, file, comment in tqdm(
    zip(s_df_filt.object_name, s_df_filt.file_name, s_df_filt.comment)
):
    draw(
        f"{root}/images_verApr2025/{source}/{file}",
        suptitle=comment,
        img_dir=s_diff_path,
    )

for source, file, comment in tqdm(
    zip(x_df_filt.object_name, x_df_filt.file_name, x_df_filt.comment)
):
    draw(
        f"{root}/images_verApr2025/{source}/{file}",
        suptitle=comment,
        img_dir=x_diff_path,
    )
