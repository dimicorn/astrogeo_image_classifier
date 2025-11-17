import argparse
import os
import pandas as pd
from yaml import safe_load
from munch import munchify
from tqdm import tqdm
from pre.visualize import draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
    )
    args = parser.parse_args()
    with open("config.yaml") as f:
        cfg = munchify(safe_load(f))
    root = cfg.data_path
    res_path = args.file.split(".")[0]
    os.makedirs(res_path, exist_ok=True)
    df = pd.read_csv(args.file)
    for source, file, comment in tqdm(zip(df.object_name, df.file_name, df.comment)):
        draw(
            f"{root}/images_verApr2025/{source}/{file}",
            suptitle=comment,
            img_dir=res_path,
        )


if __name__ == "__main__":
    main()
