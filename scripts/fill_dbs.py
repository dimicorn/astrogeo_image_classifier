from warnings import filterwarnings
import argparse
import sys
from yaml import safe_load
from munch import munchify
from utils.fill_table import FillTables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-uv",
        "--uv_table",
        type=str,
        help="SQL table with UV data, which will be created and filled",
    )
    parser.add_argument(
        "-map",
        "--map_table",
        type=str,
        help="SQL table with UV data, which will be created and filled",
    )
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    filterwarnings("ignore")

    with open("config.yaml") as f:
        cfg = munchify(safe_load(f))
    ft = FillTables(cfg)
    ft.fill(args.uv_table, args.map_table)


if __name__ == "__main__":
    main()
