import os
import json
import logging
from munch import Munch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.db import Catalogue, OurMaps
from utils.uv_fits import UVFits
from utils.map_fits import MapFits
from utils.consts import VIS_FITS
from utils.quality import quality_comment


logger = logging.getLogger(__name__)


def uv2map(file_name: str) -> str:
    return f"{file_name[:-8]}map.fits"


class FillTables(object):
    fill_table = "fill_table.log"

    def __init__(self, config: Munch, retry: bool = False) -> None:
        self.data_path, self.config_db = config.fits_path, config.db
        if not retry:
            self._get_all_files()

    def _get_all_files(self) -> tuple[dict[str : list[str]], dict[str : list[str]]]:
        objs = os.listdir(self.data_path)
        uv_files = {}
        map_files = {}
        log = open(self.fill_table, "w", encoding="utf-8")
        for obj in objs:
            uv_files[obj] = []
            map_files[obj] = []
            for file in os.listdir(f"{self.data_path}/{obj}"):
                if file[-8:] == VIS_FITS:
                    uv_files[obj].append(file)
                    log.write(f"{obj}/{file}\n")
                    map_files[obj].append(uv2map(file))
                    log.write(f"{obj}/{uv2map(file)}\n")
        log.close()
        with open("uv_files.json", "w", encoding="utf-8") as f:
            json.dump(uv_files, f)
        with open("map_files.json", "w", encoding="utf-8") as f:
            json.dump(map_files, f)
        return uv_files, map_files

    def fill(self, vis_table_name: str, map_table_name: str) -> None:
        if vis_table_name == "" or map_table_name == "":
            raise ValueError("Table names are not set")

        vis_table = Catalogue(self.config_db, vis_table_name)
        vis_table.connect2table()
        vis_table.create_table()

        map_table = OurMaps(self.config_db, map_table_name)
        map_table.connect2table()
        map_table.create_table()

        with open(self.fill_table, "r", encoding="utf-8") as f:
            n = len(f.readlines())

        with logging_redirect_tqdm():
            for _ in tqdm(range(n // 2)):
                with os.popen(f"sed -n '1,2p' {self.fill_table}") as stream:
                    block = [
                        ln.strip() for ln in stream.read().splitlines() if ln.strip()
                    ]
                    # uv_file, map_file = stream.read().rstrip().split("\n")
                if len(block) < 2:
                    logger.info("Short read; leaving last line pending")
                    break
                uv_file, map_file = block
                uv = UVFits(f"{self.data_path}/{uv_file}")
                map_ = MapFits(f"{self.data_path}/{map_file}")
                quality = quality_comment(
                    *uv.get_quality_params(), *map_.get_quality_params()
                )
                try:
                    vis_table.insert_value(uv.get_sql_params() + quality)
                    map_table.insert_value(map_.get_sql_params() + quality)
                    os.system(f"sed -i '1,2d' {self.fill_table}")
                except RuntimeError as e:
                    logger.error("%s, %s, %s", uv_file, map_file, e)
                    break
        # os.remove(self.fill_table)
