import os
from warnings import filterwarnings
import argparse
import sys
from yaml import safe_load
from munch import munchify
from utils.fill_table import FillTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-uv", "--uv_table", type=str, help="SQL table which will be created and filled"
    )
    parser.add_argument(
        "-map",
        "--map_table",
        type=str,
        help="Existing SQL table which will be used for UV data quality",
    )
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    filterwarnings("ignore")

    with open(os.getenv("CONFIG_PATH"), encoding="utf-8") as f:
        cfg = munchify(safe_load(f))
    ft = FillTable(cfg, "uv")
    ft.fill(args.uv_table, args.map_table)


if __name__ == "__main__":
    main()
