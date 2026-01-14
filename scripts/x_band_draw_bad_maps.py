import os
import argparse
from tqdm import tqdm
import pandas as pd
from yaml import safe_load
from munch import munchify
from pre.visualize import Visualize as vis


def main() -> None:
    with open(os.getenv("CONFIG_PATH")) as f:
        cfg = munchify(safe_load(f))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, help="path to directory with resulting images"
    )
    args = parser.parse_args()
    df = pd.read_csv("csv/no_good_obs.csv")
    x_band = df[df.freq_band == 'X']
    x_band = x_band.reset_index(drop=True)
    sources = x_band.object_name.tolist()

    files = []
    for source in tqdm(sources):
        path = os.path.join(cfg.fits_path, source)
        all_files = os.listdir(path)
        maps = [
            os.path.join(path, name)
            for name in all_files
            if name.lower().endswith("map.fits") and 
            'X' in name
        ]
        files.extend(maps)

    os.makedirs(args.output_path, exist_ok=True)

    for file in tqdm(files):
        if not file.endswith("map.fits"):
            print(file)
            return
        base_name = os.path.splitext(os.path.basename(file))[0]
        img_name = os.path.join(args.output_path, f"{base_name}.png")
        vis.draw_preprocessed(
            file,
            savefig=True,
            img_name=img_name
        )


if __name__ == "__main__":
    main()