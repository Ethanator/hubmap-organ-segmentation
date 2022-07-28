import numpy as np
from typing import Tuple


def mask2rle(mask: np.ndarray) -> str:
    """
    Converts a mask into its run-length encoding.

    Args:
        mask (np.ndarray): 1 - mask, 0 - background.

    Returns:
        str: Run-length encoding of the mask.
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def rle2mask(rle: str, shape: Tuple[int, int] = (1600, 256)) -> np.ndarray:
    """
    Converts a run length encoding into the corresponding mask.

    Args:
        rle (str): Run-length encoding.
        shape (Tuple[int, int], optional): (width, height) of the array to return. Defaults to (1600, 256).

    Returns:
        np.ndarray: 1 - mask, 0 - background.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape).T