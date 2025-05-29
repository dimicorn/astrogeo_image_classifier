from warnings import filterwarnings
from types import SimpleNamespace as sn
from sys import argv
import pandas as pd
from yaml import load, FullLoader
from psycopg2 import connect
from utils.beams import Beams
from utils.filter import BeamCluster
from utils.preprocess import preprocess


def drawSyntheticData(maps: pd.DataFrame, class_num: int, train_path: str = 'synt_data') -> None:
    bc = BeamCluster(maps)
    clusters = bc.beamClusterMeans(10)
    print('Finished clustering')
    print(clusters)
    b = Beams()
    beams = b.convBeams(clusters, train_path, aug=True, n=class_num)
    print(beams.shape)

def main():
    if len(argv) in [1, 2]:
        raise RuntimeError
    filterwarnings('ignore')
    with open('config/config.yaml') as f:
        config = sn(**load(f, Loader=FullLoader))
    config_db, _ = sn(**config.db), config.fits_path

    cnx = connect(
		host=config_db.host, dbname=config_db.dbname,
		user=config_db.user, password=config_db.psswd
	)
    maps = pd.read_sql(f'select * from {argv[1]};', con=cnx)

    if len(argv) == 3:
        drawSyntheticData(maps, int(argv[2]))
    else:
        drawSyntheticData(maps, int(argv[2]), argv[3])

if __name__ == '__main__':
    main()