from warnings import filterwarnings
import argparse
from yaml import safe_load
from munch import munchify
from utils.fill_table import FillTable


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('-map', '--map_table', type=str, help='SQL table which will be created and filled')
	args = parser.parse_args()
	filterwarnings('ignore')

	with open('config.yaml') as f:
		cfg = munchify(safe_load(f))
	ft = FillTable(cfg, 'map')
	ft.fill(args.map_table)

if __name__ == '__main__':
	main()
