import numpy as np


def qualityComment(
    visibilities: int,
    antennas: int,
    c_x: float,
    c_y: float,
    x: float,
    y: float,
    author: str,
    signal: float,
    noise: float,
    min_vis: int = 1000,
    min_antennas: int = 8,
    ratio: float = 10,
) -> tuple[str, str]:
    if (abs(c_x - x) > 3 or abs(c_y - y) > 3) and author != "Alan Marscher":
        dr = np.sqrt((c_x - x) * (c_x - x) + (c_y - y) * (c_y - y))
        return (0, f"distance from map center to map max {dr:.3f} pixels")
    if signal / noise <= ratio:
        return (0, f"snr = {signal / noise:.3f}")
    if visibilities < min_vis:
        return (0, f"only {visibilities} visibilities")
    if antennas < min_antennas:
        return (0, f"only {antennas} antennas")
    return (1, "")
