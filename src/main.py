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
from astrogeo.image import Image


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
		df = b.beam_cluster_means(ratio)
		# b.draw_beam_clustering(ratio)
		df.to_csv('src/astrogeo/cluster_means.csv')
		return df
	
	def draw_model_sources(self) -> None:
		b = Beams()
		b.test_beams()
	
	def draw_augmented_sources(self, clusters: pd.DataFrame = None) -> None:
		if clusters is None:
			clusters = pd.read_csv('src/astrogeo/cluster_means.csv')
		b = Beams()
		b.conv_beams(clusters, 'aug', aug=True, n=2000)
	
	def draw_astrogeo(self, maps: pd.DataFrame, path: str) -> None:
		f = Filter(maps)
		for file_name in maps.file_name:
			dir = file_name.split('_')[0]
			im = Image(f'{path}/{dir}/{file_name}')
			im.draw_map_raw('data')

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
	# uvs = pd.read_sql_table('catalogue', cnx)

	return 0

if __name__ == '__main__':
	sys.exit(main())