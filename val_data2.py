from warnings import filterwarnings
from types import SimpleNamespace as sn
from sys import argv
import os
import pandas as pd
from yaml import load, FullLoader
from psycopg2 import connect
from tqdm import tqdm
from astropy.io import fits
import numpy as np
from PIL.Image import fromarray
from utils_old.preprocess import preprocess


def drawAstrogeo(maps: pd.DataFrame, path: str, output_path: str = 'real_data') -> None:
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	for source_class, file_name in tqdm(zip(maps.source_class, maps.file_name)):
		if not os.path.isdir(f'{output_path}/{source_class}'):
			os.mkdir(f'{output_path}/{source_class}')
		dir = file_name.split('_')[0]
		name = file_name.split('.')[0]
		with fits.open(f'{path}/{dir}/{name}.fits') as f:
			im = np.array(f['PRIMARY'].data)
		im = im.squeeze()
		min_val = im.min()
		max_val = im.max()
		im = (im - min_val) / (max_val - min_val) * 255
		im_uint8 = im.astype(np.uint8)
		im_uint8 = preprocess(im_uint8)
		img = fromarray(im_uint8, mode='L')
		img.save(f'{output_path}/{source_class}/{name}.png', 'PNG')

def main():
	if len(argv) == 1:
		raise RuntimeError
	filterwarnings('ignore')
	with open('config.yaml') as f:
		config = sn(**load(f, Loader=FullLoader))
	config_db, path = sn(**config.db), config.fits_path

	cnx = connect(
		host=config_db.host, dbname=config_db.dbname,
		user=config_db.user, password=config_db.psswd
	)
	maps = pd.read_sql(f'select * from {argv[1]};', con=cnx)

	if len(argv) == 2:
		drawAstrogeo(maps, path)
	else:
		drawAstrogeo(maps, path, argv[2])


if __name__ == '__main__':
	main()