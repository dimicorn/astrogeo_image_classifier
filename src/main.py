import warnings
import yaml
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from types import SimpleNamespace as sn
import matplotlib.pyplot as plt
from astrogeo.db import FillTable
from astrogeo.filter import Filter, BeamCluster
from astrogeo.beams import Beams
from astrogeo.image import Image
from ml.train import train_model
from ml.classify import classify


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
		clusters2 = pd.read_csv('src/astrogeo/cluster_means2.csv')
		if clusters is None:
			clusters1 = pd.read_csv('src/astrogeo/cluster_means.csv')
		b = Beams()
		b.conv_beams(clusters2, 'new_aug_no_shit_noise_32k', aug=True, n=1000) # n = 2000
		# b.conv_beams(clusters2, 'new_aug_48k', aug=True, n=1000)
	
	def draw_astrogeo(self, maps: pd.DataFrame, path: str) -> None:
		f = Filter(maps)
		for file_name in maps.file_name:
			dir = file_name.split('_')[0]
			im = Image(f'{path}/{dir}/{file_name}')
			im.draw_map_raw('data_512x512_abs')
'''
def shape_hist(maps: pd.DataFrame) -> None:
	bins = np.arange(1, 3000)
	fig, ax = plt.subplots(1, 1)
	hist, bin_e = np.histogram(maps.map_size_x, bins=bins)
	ax.hist(maps.map_size_x, bins=bins)
	fig.savefig('map_x.png')
	counts_x = hist[np.where(hist > 0)]
	sizes_x = bin_e[np.where(hist > 0)]
	shape_x = dict(zip(sizes_x, counts_x))
	print(shape_x)

	fig, ax = plt.subplots(1, 1)
	hist, bin_e = np.histogram(maps.map_size_y, bins=bins)
	ax.hist(maps.map_size_y, bins=bins)
	fig.savefig('map_y.png')
	counts_y = hist[np.where(hist > 0)]
	sizes_y = bin_e[np.where(hist > 0)]
	shape_y = dict(zip(sizes_y, counts_y))
	print(shape_y)

def shape_hist_freq(maps: pd.DataFrame) -> None:
	freq_bands = {
		'L': (1, 1.8), 'S': (1.8, 2.8), 'C': (2.8, 7), 'X': (7, 9), 
        'U': (9, 17), 'K': (17, 26), 'Q': (26, 50), 'W': (50, 100), 
        'G': (100, 250)
	}
	res = {band: {'shape_x': {}, 'shape_y': {}} for band in freq_bands.keys()}
	for freq, size_x, size_y in zip(maps.freq, maps.map_size_x, maps.map_size_y):
		for band, hz in freq_bands.items():
			if hz[0] <= freq * 1e-9 <= hz[1]:
				if size_x in res[band]['shape_x']:
					res[band]['shape_x'][size_x] += 1
				else:
					res[band]['shape_x'][size_x] = 1
				
				if size_y in res[band]['shape_y']:
					res[band]['shape_y'][size_y] += 1
				else:
					res[band]['shape_y'][size_y] = 1
	print(res)
'''

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

	# a = Actions()
	# a.draw_augmented_sources(maps)

	# print(maps.noise_level.mean())
	# print(maps.noise_level.median())
	classify('cnn', 'src/ml/models/noise.pth', 
		  'src/ml/val_preds_noise.json',
		  data_path='data_512x512_abs')
	return 0

if __name__ == '__main__':
	sys.exit(main())