import os
from munch import munchify
from yaml import safe_load
from utils.uv_fits import UVFits


def test_sql_parameters():
    source = "J1608-1625"
    file = "J1608-1625_X_2014_08_09_pus_vis.fits"
    with open(os.getenv("CONFIG_PATH"), "r", encoding="utf-8") as f:
        cfg = munchify(safe_load(f))
    UVFits(f"{cfg.fits_path}/{source}/{file}")
