import warnings
import yaml
import sys
import pandas as pd
from sqlalchemy import create_engine
from types import SimpleNamespace as sn
from image import Image
from db import FillTable
from filter import Filter


def main() -> int:
	warnings.filterwarnings('ignore')
	with open('config/config.yaml') as f:
		config = sn(**yaml.load(f, Loader=yaml.FullLoader))
	config_db, path = sn(**config.db), config.path

	cnx = create_engine(f'postgresql://{config_db.user}:{config_db.psswd}@{config_db.host}/{config_db.dbname}')
	cnx.connect()
	maps, uvs = pd.read_sql_table('maps', cnx), pd.read_sql_table('catalogue', cnx)

	return 0

if __name__ == "__main__":
	res = main()
	sys.exit(res)
