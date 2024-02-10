import warnings
import yaml
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from types import SimpleNamespace as sn
from astrogeo.db import FillTable
from astrogeo.filter import Filter, BeamCluster
from astrogeo.beams import Beams


class Actions(object):
	def fill_db(self, config: sn) -> None:
		f = FillTable(config)
		f.fill_uv()
		f.fill_maps()

	def draw_filtered_maps(self, maps: pd.DataFrame, path: str) -> None:
		f = Filter(maps, 10)
		f.draw_dirty_maps(path, 'dirty_maps')
		f.draw_filtered_maps(path, 'filtered_maps')
	
	def beam_clustering(self, maps: pd.DataFrame, ratio: int) -> pd.DataFrame:
		b = BeamCluster(maps, ratio)
		# b.draw_beam_clustering(ratio)
		return b.beam_cluster_means(ratio)
	
	def draw_model_sources(self) -> None:
		shape = (128, 128)
		b = Beams(shape)
		b.test_beams()

def main() -> int:
	warnings.filterwarnings('ignore')
	with open('config/config.yaml') as f:
		config = sn(**yaml.load(f, Loader=yaml.FullLoader))
	config_db, path = sn(**config.db), config.path

	cnx = create_engine(
		f'postgresql://{config_db.user}:'
		f'{config_db.psswd}@{config_db.host}/{config_db.dbname}'
	)
	cnx.connect()
	maps = pd.read_sql_table('maps', cnx)
	uvs = pd.read_sql_table('catalogue', cnx)

	# a = Actions()
	# df = a.beam_clustering(maps, 10)
	# df.to_csv('src/astrogeo/cluster_means.csv')

	df = pd.read_csv('src/astrogeo/cluster_means.csv')
	b = Beams()
	b.conv_beams(df, 'gen')
	return 0

if __name__ == '__main__':
	sys.exit(main())