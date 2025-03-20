import os
import json
import numpy as np
from psycopg2 import connect
from psycopg2.extensions import register_adapter, AsIs
from types import SimpleNamespace as sn
from typing import Literal
from utils.fits import UVFits, MapFits
from utils.consts import MAP_FITS, VIS_FITS


register_adapter(np.float32, AsIs)
register_adapter(np.int64, AsIs)


class Table(object):
	def __init__(self, config: sn, table_name: str) -> None:
		self.host = config.host
		self.user = config.user
		self.dbname = config.dbname
		self.psswd = config.psswd
		self.table_name = table_name

		self.conn, self.cur = None, None
	
	def connect2table(self) -> None:
		with connect(
			host=self.host, dbname=self.dbname,
			user=self.user, password=self.psswd) as self.conn:
				self.cur = self.conn.cursor()

	def selectAll(self) -> list[tuple]:
		select_all = f'select * from {self.table_name};'
		self.cur.execute(select_all)
		return self.cur.fetchall()

	def dropTable(self) -> None:
		drop_query = f'drop table {self.table_name};'
		self.cur.execute(drop_query)
		self.conn.commit()
	
	def closeTable(self) -> None:
		'''Written just in case, probably will not be used'''
		if not self.conn:
			self.cur.close()
			self.conn.close()

class Catalogue(Table):
	def create_table(self) -> None:
		create_table_query = (
		f'create table if not exists {self.table_name}(\
		object_id serial primary key,\
		object_name varchar(10), obs_date varchar(15),\
		freq float, obs_author varchar(25), file_name varchar(50),\
		min_uv_radius float, max_uv_radius float,\
		visibilities int, max_amplitude float, min_amplitude float,\
		mean_amplitude float, median_amplitude float, freq_band float,\
		antenna_tables int, uv_quality varchar(100), comment varchar(100));')
		self.cur.execute(create_table_query)
		self.conn.commit()

	def insert_value(self, values: tuple) -> None:
		insert_query = (
		f'insert into {self.table_name}(object_name, obs_date, freq,\
		obs_author, file_name, min_uv_radius, max_uv_radius,\
		visibilities, max_amplitude, min_amplitude, mean_amplitude,\
		median_amplitude, freq_band, antenna_tables, uv_quality, comment)\
		values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,\
		%s, %s);')
		self.cur.execute(insert_query, values)
		self.conn.commit()

class OurMaps(Table):
	def create_table(self) -> None:
		create_table_query = (
			f'create table if not exists {self.table_name}(\
			object_id serial primary key,\
			object_name varchar(10), obs_date varchar(15),\
			freq float, obs_author varchar(25), file_name varchar(50),\
			map_max float, mapc_x float, mapc_y float,\
			map_max_x float, map_max_y float,\
			map_max_x_mas float, map_max_y_mas float,\
			noise_level float, map_size_x int, map_size_y int,\
			pixel_size_x float, pixel_size_y float, b_maj float,\
			b_min float, b_pa float, cc_tables int,\
			map_quality int, comment varchar(100));'
		)
		self.cur.execute(create_table_query)
		self.conn.commit()
	
	def insert_value(self, values: tuple) -> None:
		insert_query = (
			f'insert into {self.table_name}(object_name, obs_date, freq,\
			obs_author, file_name, map_max, mapc_x, mapc_y,\
			map_max_x, map_max_y, map_max_x_mas, map_max_y_mas,\
			noise_level, map_size_x, map_size_y, pixel_size_x, pixel_size_y,\
			b_maj, b_min, b_pa, cc_tables, map_quality, comment)\
			values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
			%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);'
		)
		self.cur.execute(insert_query, values)
		self.conn.commit()

class FillTable(object):
	master_maps = 'master_maps.txt'
	master_uvs = 'master_uvs.txt'
	
	def __init__(self, config: sn, table_type: Literal['uv', 'map']) -> None:
		self.data_path, self.config_db = config.fits_path, sn(**config.db)
		self.table_type = table_type
		self._getAllFiles()
	
	def _getAllFilesUV(self) -> None:
		objs = os.listdir(self.data_path)
		uv_files = {}
		uv = open(self.master_uvs, 'w')
		for obj in objs:
			uv_files[obj] = []
			for file in os.listdir(f'{self.data_path}/{obj}'):
				if file[-8:] == VIS_FITS:
					uv_files[obj].append(file)
					uv.write(f'{obj}/{file}\n')
		uv.close()
		with open('uv_files.json', 'w') as f:
			json.dump(uv_files, f)
		return uv_files
	
	def _getAllFilesMaps(self) -> None:
		objs = os.listdir(self.data_path)
		map_files = {}
		m = open(self.master_maps, 'w')
		for obj in objs:
			map_files[obj] = []
			for file in os.listdir(f'{self.data_path}/{obj}'):
				if file[-8:] == MAP_FITS:
					map_files[obj].append(file)
					m.write(f'{obj}/{file}\n')
		m.close()
	
		with open('map_files.json', 'w') as f:
			json.dump(map_files, f)
	
	def _getAllFiles(self) -> None:
		if self.table_type == 'uv':
			self._getAllFilesUV()
		self._getAllFilesMaps()
	
	def fill(self, name: str) -> None:
		if self.table_type == 'uv':
			self._fillUV(name)
		self._fillMaps(name)
	
	def _fillUV(self, name: str) -> None:
		table = Catalogue(self.config_db, name)
		table.connect2table()
		table.create_table()
		with open(self.master_uvs, 'r') as f:
			n = len(f.readlines())
		
		for _ in range(n):
			file = os.popen(f"sed -n '1p' {self.master_uvs}").read().rstrip()
			uv = UVFits(f'{self.data_path}/{file}')
			table.insert_value(uv.getSQLParams())
			os.system(f"sed -i '1d' {self.master_uvs}")

	def _fillMaps(self, name: str) -> None:
		table = OurMaps(self.config_db, name)
		table.connect2table()
		table.create_table()
		with open(self.master_maps, 'r') as f:
			n = len(f.readlines())
		
		for _ in range(n):
			file = os.popen(f"sed -n '1p' {self.master_maps}").read().rstrip()
			map = MapFits(f'{self.data_path}/{file}')
			table.insert_value(map.getSQLParams())
			os.system(f"sed -i '1d' {self.master_maps}")