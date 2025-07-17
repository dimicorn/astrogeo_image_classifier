from warnings import filterwarnings
from yaml import safe_load
from munch import munchify
from utils.fill_table import FillTable


def main() -> None:
	filterwarnings('ignore')
	with open('config.yaml') as f:
		cfg = munchify(safe_load(f))
	ft = FillTable(cfg, 'uv')
	ft.fill('uvs_Apr25_test', 'maps_Apr25_test')

if __name__ == '__main__':
	main()
