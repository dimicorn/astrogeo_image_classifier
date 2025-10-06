import os
import json
from typing import Literal
from utils.db import Catalogue, OurMaps
from utils.map_fits import MapFits
from utils.uv_fits import UVFits
from utils.consts import VIS_FITS, MAP_FITS
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class FillTable(object):
    master_maps = "master_maps.txt"
    master_uvs = "master_uvs.txt"

    def __init__(self, config, table_type: Literal["uv", "map"]) -> None:
        self.data_path, self.config_db = config.fits_path, config.db
        self.table_type = table_type
        self._getAllFiles()

    def _getAllFilesUV(self) -> None:
        objs = os.listdir(self.data_path)
        uv_files = {}
        uv = open(self.master_uvs, "w")
        for obj in objs:
            uv_files[obj] = []
            for file in os.listdir(f"{self.data_path}/{obj}"):
                if file[-8:] == VIS_FITS:
                    uv_files[obj].append(file)
                    uv.write(f"{obj}/{file}\n")
        uv.close()
        with open("uv_files.json", "w") as f:
            json.dump(uv_files, f)
        return uv_files

    def _getAllFilesMaps(self) -> None:
        objs = os.listdir(self.data_path)
        map_files = {}
        m = open(self.master_maps, "w")
        for obj in objs:
            map_files[obj] = []
            for file in os.listdir(f"{self.data_path}/{obj}"):
                if file[-8:] == MAP_FITS:
                    map_files[obj].append(file)
                    m.write(f"{obj}/{file}\n")
        m.close()

        with open("map_files.json", "w") as f:
            json.dump(map_files, f)

    def _getAllFiles(self) -> None:
        if self.table_type == "uv":
            self._getAllFilesUV()
        elif self.table_type == "map":
            self._getAllFilesMaps()

    def fill(self, name: str, map_table_name: str = None) -> None:
        if self.table_type == "uv":
            self._fillUV(name, map_table_name)
        elif self.table_type == "map":
            self._fillMaps(name)

    def _fillUV(self, name: str, map_table_name: str) -> None:
        if map_table_name is None:
            raise ValueError("Map table is not set")
        table = Catalogue(self.config_db, name)
        table.connect2table()
        table.create_table()
        with open(self.master_uvs, "r") as f:
            n = len(f.readlines())

        with logging_redirect_tqdm():
            for _ in tqdm(range(n)):
                file = os.popen(f"sed -n '1p' {self.master_uvs}").read().rstrip()
                uv = UVFits(f"{self.data_path}/{file}")
                table.insert_value(uv.getSQLParams(map_table_name, self.config_db))
                os.system(f"sed -i '1d' {self.master_uvs}")

    def _fillMaps(self, name: str) -> None:
        table = OurMaps(self.config_db, name)
        table.connect2table()
        table.create_table()
        with open(self.master_maps, "r") as f:
            n = len(f.readlines())
        with logging_redirect_tqdm():
            for _ in tqdm(range(n)):
                file = os.popen(f"sed -n '1p' {self.master_maps}").read().rstrip()
                map = MapFits(f"{self.data_path}/{file}")
                table.insert_value(map.getSQLParams())
                os.system(f"sed -i '1d' {self.master_maps}")
