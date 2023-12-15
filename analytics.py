import yaml
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def signal_noise(df: pd.DataFrame):
	signal_noise = [sig/nl for sig, nl in zip(df.map_max, df.noise_level)]
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_title('Signal/noise distribution')
	sns.histplot(signal_noise)
	ax.set_xlabel('Signal-noise ratio')
	ax.set_ylabel('Number')
	plt.savefig('test/signal_noise_dist.png', dpi=500)
	plt.close(fig)

def weird_maps(df: pd.DataFrame):
	weird_maps = []
	for c_x, x, c_y, y, a, file in zip(df.mapc_x, df.map_max_x, df.mapc_y, df.map_max_y, df.obs_author, df.file_name):
		if (abs(c_x - x) > 3 or abs(c_y - y) > 3) and a != 'Alan Marscher':
			weird_maps.append(file)
	return weird_maps

def main():
	with open('config.yaml') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	config_db = config['db']

	cnx = create_engine(f"postgresql://{config_db['user']}:{config_db['psswd']}@{config_db['host']}/{config_db['dbname']}")
	cnx.connect()
	maps = pd.read_sql_table('maps', cnx)
	uvs = pd.read_sql_table('catalogue', cnx)


if __name__ == '__main__':
	main()
