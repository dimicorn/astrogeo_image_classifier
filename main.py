import os
import sys
import warnings
import yaml
from image import Image
from db import Catalogue
from consts import *


def database() -> None:
	with open('config.yaml') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)['db']

	t = Catalogue(config, 'catalogue')
	t.connect2table()
	t.create_table()
	t.select_all()

def main() -> int:
	warnings.filterwarnings('ignore')
	
	with open('config.yaml') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
	data_path = config['path']
	# obj = os.listdir(data_path)[1]
	# obj = 'J2212+2355'
	
	obj = 'J0725+1032'
	uv_files = [file for file in os.listdir(f'{data_path}/{obj}') 
	            if file[-8:] == VIS_FITS]
	# map_files = [file for file in os.listdir(f'{data_path}/{obj}') 
	# 			 if file[-8:] == MAP_FITS]
	test = Image(f'{data_path}/{obj}/{uv_files[0]}')
	print(uv_files[0])

	test.draw_uv()
	test.draw_phase_radius()
	test.draw_ampl_radius()
	test.draw_uv_lof()
	test.draw_uv_3d()

	# test.draw_map()
	'''
	objs = os.listdir(data_path)
	for obj in objs:
		map_files = [file for file in os.listdir(f'{data_path}/{obj}')
					 if file[-8:] == MAP_FITS]
		test = Image(f'{data_path}/{obj}/{map_files[0]}')
		print(map_files[0])
		test.draw_map()
	'''
	return 0

if __name__ == "__main__":
	res = main()
	sys.exit(res)
