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

def mean_clusters() -> pd.DataFrame:
	b_maj = [4.15e+01, 1.19e+01, 1.79e+01, 2.18e+01, 1.53e+01, 1.68e+01, 1.48e+01, 1.69e+01, 2.19e+02, 9.17e+01]
	b_min = [1.26e+01, 5.44e+00, 8.17e+00, 8.31e+00, 7.96e+00, 7.11e+00, 6.53e+00, 9.08e+00, 3.78e+01, 2.23e+01]
	b_pa = [-8.5e+00, -2.92e+00, 3.58e+01, -6.84e+00, -6.61e+01, 1.38e+01, -2.38e+01, 7.17e+01, -2.78e+00, 5.76e+00]
	amount = [7777, 54997, 6147, 17691, 3834, 17499, 11545, 3248, 203, 1779]
	means = {'b_maj': b_maj, 'b_min': b_min, 'b_pa': b_pa, 'amount': amount}
	cluster_means = pd.DataFrame(means)
	cluster_means.index.name = 'Cluster'
	return cluster_means

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

	return 0

if __name__ == '__main__':
	sys.exit(main())