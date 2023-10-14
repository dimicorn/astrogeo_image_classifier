import os
import sys
import warnings
import yaml
from image import Image
from db import Catalogue
from consts import *


def database():
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
    # obj = os.listdir(data_path)[0]
    obj = 'J2212+2355'
    uv_files = [file for file in os.listdir(f'{data_path}/{obj}') 
                if file[-8:] == VIS_FITS]
    # map_files = [file for file in os.listdir(f'{data_path}/{obj}') 
    #              if file[-8:] == MAP_FITS]
    test = Image(f'{data_path}/{obj}/{uv_files[0]}')
    # print(f'{data_path}/{obj}/{map_files[1]}')
    # test.print_header()
    test.draw_uv()
    # test.draw_map()
    # print(test.get_models())
    test.draw_uv_lof()
    # test.print_header()

    return 0

if __name__ == "__main__":
    res = main()
    sys.exit(res)
