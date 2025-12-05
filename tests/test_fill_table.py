import os
from yaml import safe_load
from munch import munchify
from utils.fill_table import uv2map, FillTables


def test_uv2map():
    uv_name = "J1608-1625_X_2014_08_09_pus_vis.fits"
    map_name = uv2map(uv_name)
    assert isinstance(map_name, str)
    assert map_name == "J1608-1625_X_2014_08_09_pus_map.fits"


def test_get_all_files():
    with open(os.getenv("CONFIG_PATH"), "r", encoding="utf-8") as f:
        cfg = munchify(safe_load(f))
    ft = FillTables(cfg)
    uvs, maps = ft._get_all_files()  # pylint: disable=protected-access
    assert isinstance(uvs, dict) and isinstance(maps, dict)
    assert len(uvs) == len(maps)
    for obj_uv, obj_map in zip(uvs, maps):
        assert isinstance(obj_uv, str) and isinstance(obj_map, str)
        assert isinstance(uvs[obj_uv], list) and isinstance(maps[obj_map], list)
        assert obj_uv == obj_map
        for obs_uv, obs_map in zip(uvs[obj_uv], maps[obj_map]):
            assert isinstance(obs_uv, str) and isinstance(obs_map, str)
            assert uv2map(obs_uv) == obs_map


def test_logFiles():
    assert os.path.exists("fill_table.log")
    assert os.path.exists("uv_files.json")
    assert os.path.exists("map_files.json")
    os.remove("fill_table.log")
    os.remove("uv_files.json")
    os.remove("map_files.json")
