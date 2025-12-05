import os
from munch import munchify
from yaml import safe_load
from utils.map_fits import MapFits


def test_sql_parameters():
    source = "J1608-1625"
    file = "J1608-1625_X_2014_08_09_pus_map.fits"
    with open(os.getenv("CONFIG_PATH"), "r", encoding="utf-8") as f:
        cfg = munchify(safe_load(f))
    mp = MapFits(f"{cfg.fits_path}/{source}/{file}")
    res = mp.get_sql_params()
    assert len(res) == 21
    (
        object_name,
        obs_date,
        freq,
        obs_author,
        file_name,
        map_max,
        mapc_x,
        mapc_y,
        map_max_x,
        map_max_y,
        map_max_x_mas,
        map_max_y_mas,
        noise_level,
        map_size_x,
        map_size_y,
        pixel_size_x,
        pixel_size_y,
        b_maj,
        b_min,
        b_pa,
        cc_tables,
    ) = res
    assert isinstance(object_name, str)
    assert len(object_name) <= 10
    assert isinstance(obs_date, str)
    assert len(obs_date) <= 15
    assert isinstance(freq, float)
    assert isinstance(obs_author, str)
    assert len(obs_author) <= 25
    assert isinstance(file_name, str)
    assert len(file_name) < 50
    assert isinstance(map_max, float)
    assert isinstance(mapc_x, float)
    assert isinstance(mapc_y, float)
    assert isinstance(map_max_x, float)
    assert isinstance(map_max_y, float)
    assert isinstance(map_max_x_mas, float)
    assert isinstance(map_max_y_mas, float)
    assert isinstance(noise_level, float)
    assert isinstance(map_size_x, int)
    assert isinstance(map_size_y, int)
    assert isinstance(pixel_size_x, float)
    assert isinstance(pixel_size_y, float)
    assert isinstance(b_maj, float)
    assert isinstance(b_min, float)
    assert isinstance(b_pa, float)
    assert isinstance(cc_tables, int)
