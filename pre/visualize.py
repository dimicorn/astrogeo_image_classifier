import re
import numpy as np
import matplotlib.pyplot as plt
from .load import fits2numpy
from .preprocess import preprocess, preprocess_lognorm


def getLabel(im: np.ndarray) -> str:
    label = (
        f"Max: {im.max():.2f}, Min: {im.min():.3f}, Sum: {im.sum():.2f}\n"
        f"Height: {im.shape[0]}, Width: {im.shape[1]}"
    )
    return label


def showProgress(
    filename: str,
    raw_image: np.ndarray,
    image_preproc: np.ndarray,
    image_lognorm: np.ndarray,
    suptitle: str = None,
    img_dir: str = None,
    ax=None,
    xlabel: str = None,
    title: str = None,
) -> None:
    # TODO: horizontal flip
    if ax is not None and xlabel is not None and title is not None:
        xlabel = re.sub(r"(\d+\.\d+)", lambda m: f"{float(m.group()):.1f}", xlabel)
        ax.imshow(raw_image, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if img_dir is None:
        print("no img_dir was set")
        return

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    if suptitle is not None:
        fig.suptitle(suptitle)
    xlabel_raw = getLabel(raw_image)
    axs[0].imshow(raw_image, cmap="gray")
    axs[0].set_title("Raw map")
    axs[0].set_xlabel(xlabel_raw)

    xlabel_preproc = getLabel(image_preproc)
    axs[1].imshow(image_preproc, cmap="gray")
    axs[1].set_title("Preprocessed map")
    axs[1].set_xlabel(xlabel_preproc)

    xlabel_lognorm = getLabel(image_lognorm)
    axs[2].imshow(image_lognorm, cmap="gray")
    axs[2].set_title("Lognorm map")
    axs[2].set_xlabel(xlabel_lognorm)

    axs[3].hist(image_lognorm.ravel(), bins=100)
    axs[3].set_title("Histogram of log-normalized values")
    axs[3].set_yscale("log")
    axs[3].grid(True)
    axs[3].set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(f"{img_dir}/{filename}")


def draw(
    fits_path: str,
    img_dir: str = None,
    suptitle: str = None,
    npy: bool = False,
    npy_path: str = None,
    npy_log_path: str = None,
    ax=None,
    xlabel: str = None,
    title: str = None,
) -> None:
    raw_im = fits2numpy(fits_path)
    im = preprocess(raw_im)
    im_lognorm = preprocess_lognorm(raw_im)
    filename = fits_path.split("/")[-1].split(".")[0]
    if npy and npy_log_path is not None and npy_path is not None:
        np.save(f"{npy_path}/{filename}", im)
        np.save(f"{npy_log_path}/{filename}_lognorm", im_lognorm)
    showProgress(
        filename,
        raw_im,
        im,
        im_lognorm,
        suptitle=suptitle,
        img_dir=img_dir,
        ax=ax,
        xlabel=xlabel,
        title=title,
    )
