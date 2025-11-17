from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from munch import munchify
from yaml import safe_load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pre.visualize import draw


def main():
    with open("config.yml") as f:
        cfg = munchify(safe_load(f))

    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to csv file with data")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    with PdfPages(args.output) as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(10, 15))
        count = 0
        for object_name, file_name, comment in tqdm(
            zip(df.object_name, df.file_name, df.comment), total=len(df.object_name)
        ):
            ax = axes[count // 3][count % 3]

            fig = draw(
                f"{cfg.fits_path}/{object_name}/{file_name}",
                suptitle=comment,
                ax=ax,
                xlabel=comment,
                title=file_name,
            )
            count += 1
            if count == 9:
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # start new page
                fig, axes = plt.subplots(3, 3, figsize=(10, 15))
                count = 0

        if count > 0:
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
