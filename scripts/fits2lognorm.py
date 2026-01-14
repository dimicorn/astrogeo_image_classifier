import os
import argparse
from tqdm import tqdm
from pre.visualize import Visualize as vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-path", type=str, help="path to directory with fits data"
    )
    parser.add_argument(
        "--output-path", type=str, help="path to directory with resulting images"
    )
    args = parser.parse_args()
    files = os.listdir(args.dir_path)
    
    os.makedirs(args.output_path, exist_ok=True)

    for file in tqdm(files):
        base_name = os.path.splitext(os.path.basename(file))[0]
        img_name = os.path.join(args.output_path, f"{base_name}.png")
        vis.draw_lognorm(
            os.path.join(args.dir_path, file),
            savefig=True,
            img_name=img_name
        )


if __name__ == "__main__":
    main()