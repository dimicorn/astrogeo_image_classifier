import numpy as np
import matplotlib.pyplot as plt


def getLabel(im: np.ndarray) -> str:
    label = (
        f'Max: {im.max():.2f}, Min: {im.min():.3f}, Sum: {im.sum():.2f}\n'
        f'Height: {im.shape[0]}, Width: {im.shape[1]}'
    )
    return label

def showProgress(filename: str, raw_image: np.ndarray, image_preproc: np.ndarray, image_lognorm: np.ndarray) -> None:
    # TODO: horizontal flip
    _, axs = plt.subplots(1, 4, figsize=(15, 5))
    xlabel_raw = getLabel(raw_image)
    axs[0].imshow(raw_image, cmap='gray')
    axs[0].set_title('Raw map')
    axs[0].set_xlabel(xlabel_raw)

    xlabel_preproc = getLabel(image_preproc)
    axs[1].imshow(image_preproc, cmap='gray')
    axs[1].set_title('Preprocessed map')
    axs[1].set_xlabel(xlabel_preproc)

    xlabel_lognorm = getLabel(image_lognorm)
    axs[2].imshow(image_lognorm, cmap='gray')
    axs[2].set_title('Lognorm map')
    axs[2].set_xlabel(xlabel_lognorm)

    axs[3].hist(image_lognorm.ravel(), bins=100)
    axs[3].set_title('Histogram of log-normalized values')
    axs[3].set_yscale('log')
    axs[3].grid(True)
    axs[3].set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(filename)