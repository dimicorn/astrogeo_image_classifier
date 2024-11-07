import warnings
from yaml import load, FullLoader
import sys
import pandas as pd
from types import SimpleNamespace as sn
from astrogeo.db import FillTable
from astrogeo.filter import Filter, BeamCluster
from astrogeo.beams import Beams
from astrogeo.image import Image
from psycopg2 import connect


class Actions(object):
	def fill_db(self, config: sn) -> None:
		f = FillTable(config)
		f.fill_uv('catalogue24')
		# f.fill_maps('maps24')

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
		clusters2 = pd.read_csv('src/astrogeo/cluster_means2.csv')
		# if clusters is None:
		# 	clusters1 = pd.read_csv('src/astrogeo/cluster_means.csv')
		b = Beams()
		# b.conv_beams(clusters2, 'noise_96k', aug=True, n=3000)
		b.conv_beams(clusters2, 'synt_one_channel_test', aug=True, n=1)
	
	def draw_astrogeo(self, maps: pd.DataFrame, path: str) -> None:
		# f = Filter(maps)
		for quality, file_name in zip(maps.map_quality, maps.file_name):
			if quality == 1:
				dir = file_name.split('_')[0]
				im = Image(f'{path}/{dir}/{file_name}')
				im.draw_map_raw('data_one_channel')

def main() -> int:
	warnings.filterwarnings('ignore')
	with open('config/config.yaml') as f:
		config = sn(**load(f, Loader=FullLoader))
	config_db, path = sn(**config.db), config.fits_path

	cnx = connect(
		host=config_db.host, dbname=config_db.dbname,
		user=config_db.user, password=config_db.psswd
	)
	maps = pd.read_sql('select * from maps24;', con=cnx)
	# uvs = pd.read_sql_table('catalogue24', cnx)

	a = Actions()
	# a.draw_astrogeo(maps, path)
	# a.draw_augmented_sources()
	return 0

if __name__ == '__main__':
	sys.exit(main())