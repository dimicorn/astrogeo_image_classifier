import os
import sys
import warnings
import yaml
import json
from fits import UVFits, MapFits
from db import Catalogue, OurMaps
from consts import *


class FillTable:
	master_maps, master_uvs = 'master_maps.txt', 'master_uvs.txt'
	def __init__(self) -> None:
		with open('config.yaml') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		self.data_path, self.config_db = config['path'], config['db']
		self.maps, self.uvs = self.get_all_files()
	
	def get_all_files(self) -> tuple:
		objs = os.listdir(self.data_path)
		map_files, uv_files = {}, {}
		m = open(self.master_maps, 'w')
		uv = open(self.master_uvs, 'w')
		for obj in objs:
			map_files[obj], uv_files[obj] = [], []
			for file in os.listdir(f'{self.data_path}/{obj}'):
				if file[-8:] == MAP_FITS:
					map_files[obj].append(file)
					m.write(f'{obj}/{file}\n')
				elif file[-8:] == VIS_FITS:
					uv_files[obj].append(file)
					uv.write(f'{obj}/{file}\n')
		m.close()
		uv.close()
	
		with open('map_files.json', 'w') as f:
			json.dump(map_files, f)
	
		with open('uv_files.json', 'w') as f:
			json.dump(uv_files, f)
		return (map_files, uv_files)
	
	def fill_uv(self) -> None:
		table = Catalogue(self.config_db, 'catalogue')
		table.connect2table()
		table.create_table()
		with open(self.master_uvs, 'r') as f:
			n = len(f.readlines())
		
		for _ in range(n):
			file = os.popen(f"sed -n '1p' {self.master_uvs}").read().rstrip()
			uv = UVFits(f'{self.data_path}/{file}')
			table.insert_value(uv.get_sql_params())
			os.system(f"sed -i '1d' {self.master_uvs}")

	def fill_maps(self) -> None:
		table = OurMaps(self.config_db, 'maps')
		table.connect2table()
		table.create_table()
		with open(self.master_maps, 'r') as f:
			n = len(f.readlines())
		
		for _ in range(n):
			file = os.popen(f"sed -n '1p' {self.master_maps}").read().rstrip()
			map = MapFits(f'{self.data_path}/{file}')
			table.insert_value(map.get_sql_params())
			os.system(f"sed -i '1d' {self.master_maps}")

def main() -> int:
	warnings.filterwarnings('ignore')
	# f = FillTable()
	# f.get_all_files()
	# f.fill_uv()

	return 0

if __name__ == "__main__":
	res = main()
	sys.exit(res)
