from warnings import filterwarnings
from types import SimpleNamespace as sn
from sys import argv
import pandas as pd
from yaml import load, FullLoader
from psycopg2 import connect
from tqdm import tqdm
from utils_old.image import Image
from utils_old.filter import Filter
from utils_old.preprocess import preprocess


def drawAstrogeo(maps: pd.DataFrame, path: str, val_path: str = "real_data") -> None:
    Filter(maps)
    print("Finished filtering")
    for quality, file_name in tqdm(zip(maps.map_quality, maps.file_name)):
        if quality == 1:
            dir = file_name.split("_")[0]
            im = Image(f"{path}/{dir}/{file_name}")
            map2d = im.mapData().squeeze()
            map2d = preprocess(map2d)
            im.drawMapRaw(val_path)


def main():
    if len(argv) == 1:
        raise RuntimeError
    filterwarnings("ignore")
    with open("config.yaml") as f:
        config = sn(**load(f, Loader=FullLoader))
    config_db, path = sn(**config.db), config.fits_path

    cnx = connect(
        host=config_db.host,
        dbname=config_db.dbname,
        user=config_db.user,
        password=config_db.psswd,
    )
    maps = pd.read_sql(f"select * from {argv[1]};", con=cnx)

    if len(argv) == 2:
        drawAstrogeo(maps, path)
    else:
        drawAstrogeo(maps, path, argv[2])


if __name__ == "__main__":
    main()
