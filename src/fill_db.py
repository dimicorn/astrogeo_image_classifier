from yaml import load, FullLoader
from sys import argv
from types import SimpleNamespace as sn
from warnings import filterwarnings
from typing import Literal
from utils.db import FillTable


def fill_db(config: sn, table_type: Literal['uv', 'map'], db_name: str = 'test_table') -> None:
    if table_type not in ['uv', 'map']:
        raise RuntimeError
    f = FillTable(config, table_type)
    f.fill(db_name)

def main():
    if len(argv) == 1:
        raise RuntimeError
    filterwarnings('ignore')
    with open('config/config.yaml') as f:
        config = sn(**load(f, Loader=FullLoader))
    if len(argv) == 2:
        fill_db(config, table_type=argv[1])
    else:
        fill_db(config, table_type=argv[1], db_name=argv[2])

if __name__ == '__main__':
    main()