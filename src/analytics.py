import warnings
import yaml
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from image import Image
import os
from consts import *
from types import SimpleNamespace as sn


def signal_noise_dist(df: pd.DataFrame) -> None:
	signal_noise = [sig/nl for sig, nl in zip(df.map_max, df.noise_level)]
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_title('Signal/noise distribution')
	sns.histplot(signal_noise)
	ax.set_xlabel('Signal-noise ratio')
	ax.set_ylabel('Number')
	plt.savefig('test/signal_noise_dist.png', dpi=500)
	plt.close(fig)

def weird_maps(df: pd.DataFrame) -> list:
	weird_maps = []
	for c_x, x, c_y, y, a, file in zip(df.mapc_x, df.map_max_x, df.mapc_y, df.map_max_y, df.obs_author, df.file_name):
		if (abs(c_x - x) > 3 or abs(c_y - y) > 3) and a != 'Alan Marscher':
			weird_maps.append(file)
	return weird_maps

def main() -> None:
	warnings.filterwarnings('ignore')
	with open('config/config.yaml') as f:
		config = sn(**yaml.load(f, Loader=yaml.FullLoader))
	config_db = sn(**config.db)
	path = config.path
	
	# obj = 'J1657+4808'
	# file = 'J1657+4808_S_2017_04_28_pet_map.fits'
	# obj = 'J1229+0203'
	# file = 'J1229+0203_Q_2009_07_26_mar_map.fits'
	# im = Image(f'{path}/{obj}/{file}')
	# im.draw_map()

	cnx = create_engine(f'postgresql://{config_db.user}:{config_db.psswd}@{config_db.host}/{config_db.dbname}')
	cnx.connect()
	maps = pd.read_sql_table('maps', cnx)
	uvs = pd.read_sql_table('catalogue', cnx)
	wm = set(weird_maps(maps))
	signal_noise = {file: sig/nl 
				 for file, sig, nl in zip(maps.file_name, maps.map_max, maps.noise_level) if file in wm}
	sn_5 = [key for key, val in signal_noise.items() if val <= 5]
	sn_10 = [key for key, val in signal_noise.items() if val <= 10]

	print(f'Number of object signal-noise ratio <= 5: {len(sn_5)}')
	print(f'Number of object signal-noise ratio <= 10: {len(sn_10)}')

	# for file in res:
	# 	folder = file.split('_')[0]
	# 	im = Image(f'{path}/{folder}/{file}')
	# 	im.draw_map()

	# fig = plt.figure(figsize=(10, 8))
	# ax = fig.add_subplot(1, 1, 1)
	# ax.set_title('Signal/noise distribution')
	# sns.histplot(signal_noise, bins=2500)
	# ax.set_xlabel('Signal-noise ratio')
	# ax.set_ylabel('Number')
	# ax.set_xlim(0, 50)
	# plt.savefig('test/signal_noise_dist2.png', dpi=500)
	# plt.close(fig)


if __name__ == '__main__':
	main()
