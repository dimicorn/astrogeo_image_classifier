import sys
import yaml
from image import Image
from db import Catalogue


def main() -> int:
    im = Image()
    obj = im.get_objects()[0]
    im.draw_map(obj)
    im.draw_uv(obj)
    # print(im.get_parameters(obj))

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['db']

    t = Catalogue(config, 'catalogue')
    t.connect2table()
    t.create_table()
    t.select_all()

    return 0

if __name__ == "__main__":
    res = main()
    sys.exit(res)
