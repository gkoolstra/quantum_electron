import numpy as np

def construct_symmetric_y(ymin, N):
    """
    This helper function constructs a one-sided array from ymin to -dy/2 with N points.
    The spacing is chosen such that, when mirrored around y = 0, the spacing is constant.

    This requirement limits our choice for dy, because the spacing must be such that there's
    an integer number of points in yeval. This can only be the case if
    dy = 2 * ymin / (2*k+1) and Ny = ymin / dy - 0.5 + 1
    yeval = y0, y0 - dy, ... , -3dy/2, -dy/2
    :param ymin: Most negative value
    :param N: Number of samples in the one-sided array
    :return: One-sided array of length N.
    """
    dy = 2 * np.abs(ymin) / np.float(2 * N + 1)
    return np.linspace(ymin, -dy / 2., int((np.abs(ymin) - 0.5 * dy) / dy + 1))