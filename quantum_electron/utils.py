import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
import pyvista
from shapely import Polygon
import shapely.plotting
from matplotlib import pyplot as plt
from scipy.constants import elementary_charge as qe, epsilon_0
from scipy.constants import Boltzmann as kB

def select_outer_electrons(xi: ArrayLike, yi: ArrayLike, plot: bool=True, **kwargs) -> tuple:
    """Select the outermost electrons from a small ensemble of electrons. This is 
    useful for calculating the area of an ensemble.

    Args:
        xi (ArrayLike): electron x-positions np.array([x0, x1, ...])
        yi (ArrayLike): electron y-positions np.array([y0, y1, ...])
        plot (bool, optional): Plot the polygon. Defaults to True.

    Returns:
        tuple: Polygon points (x and y), polygon area
    """
    # There must be at least 2 electrons to define a surface
    if len(xi) > 2: 
        points = np.c_[xi.reshape(-1), yi.reshape(-1), np.zeros(len(yi)).reshape(-1)]
        cloud = pyvista.PolyData(points)
        surf = cloud.delaunay_2d()
        boundary = surf.extract_feature_edges(boundary_edges=True, 
                                              non_manifold_edges=False, 
                                              manifold_edges=False)

        boundary_x = boundary.points[:, 0] * 1e6
        boundary_y = boundary.points[:, 1] * 1e6

        # Calculate the center of mass x and y coordinates.
        x_com = np.mean(boundary_x)
        y_com = np.mean(boundary_y)

        # Order the electrons clockwise to draw the polygon correctly.
        angles = np.arctan2((boundary_y - y_com), (boundary_x - x_com))
        boundary_x = boundary_x[np.argsort(angles)]
        boundary_y = boundary_y[np.argsort(angles)]

        boundary_array = xy2r(boundary_x, boundary_y).reshape(-1, 2)
        polygon = Polygon(boundary_array)

        if plot:
            shapely.plotting.plot_polygon(polygon, **kwargs)
            plt.grid(None)
    
        return polygon.exterior.xy, polygon.area
    else:
        return None, None
        
    
def density_from_positions(xi: ArrayLike, yi: ArrayLike) -> float:
    """Electron density estimate calculated from the nearest neighbor distance

    Args:
        xi (ArrayLike): electron x-positions np.array([x0, x1, ...])
        yi (ArrayLike): electron y-positions np.array([y0, y1, ...])

    Returns:
        float: Electron density in units of m^-2
    """
    Xi, Yi = np.meshgrid(xi, yi)
    Xj, Yj = Xi.T, Yi.T

    XiXj = Xi - Xj
    YiYj = Yi - Yj

    Rij_standard = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
    np.fill_diagonal(Rij_standard, np.inf)

    nearest_neighbor_distance = np.min(Rij_standard, axis=1)
    area = np.pi * np.mean(nearest_neighbor_distance) ** 2 / 4
    return 1 / area

def mean_electron_spacing(xi: ArrayLike, yi: ArrayLike) -> float:
    """Mean electron spacing calculated from the nearest neighbor distance

    Args:
        xi (ArrayLike): electron x-positions np.array([x0, x1, ...])
        yi (ArrayLike): electron y-positions np.array([y0, y1, ...])

    Returns:
        float: Mean electron spacing in units of m
    """
    Xi, Yi = np.meshgrid(xi, yi)
    Xj, Yj = Xi.T, Yi.T

    XiXj = Xi - Xj
    YiYj = Yi - Yj

    Rij_standard = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
    np.fill_diagonal(Rij_standard, np.inf)

    nearest_neighbor_distance = np.min(Rij_standard, axis=1)
    return np.mean(nearest_neighbor_distance)

def gamma_parameter(xi: ArrayLike, yi: ArrayLike, T: float) -> float:
    """Ratio of the Coulomb energy to kinetic energy. For bulk electrons on helium 
    the critical value is 137. If the value exceeds the critical value, we have a Wigner solid. 
    For values below the critical value we have a liquid.

    Args:
        xi (ArrayLike): electron x-positions np.array([x0, x1, ...])
        yi (ArrayLike): electron y-positions np.array([y0, y1, ...])
        T (float): Temperature

    Returns:
        float: Ratio of the Coulomb energy to the Kinetic energy
    """
    nearest_neighbor_distance = 1 / np.sqrt(np.pi * density_from_positions(xi, yi))    
    return qe ** 2 / (4 * np.pi * epsilon_0 * nearest_neighbor_distance)  / (kB * T)
    
def construct_symmetric_y(ymin: float, N: int) -> ArrayLike:
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
    dy = 2 * np.abs(ymin) / float(2 * N + 1)
    return np.linspace(ymin, -dy / 2., int((np.abs(ymin) - 0.5 * dy) / dy + 1))

def find_nearest(array: ArrayLike, value: float) -> int:
    """
    Finds the nearest value in array. Returns index of array for which this is true.
    """
    idx=(np.abs(array-value)).argmin()
    return int(idx)

def r2xy(r: ArrayLike) -> tuple:
    """
    Reformat electron position array.
    :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
    :return: np.array([x0, x1, ...]), np.array([y0, y1, ...])
    """
    return r[::2], r[1::2]

def xy2r(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Reformat electron position array.
    :param x: np.array([x0, x1, ...])
    :param y: np.array([y0, y1, ...])
    :return: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
    """
    if len(x) == len(y):
        r = np.zeros(2 * len(x))
        r[::2] = x
        r[1::2] = y
        return r
    else:
        raise ValueError("x and y must have the same length!")
    
def crop_potential(x: ArrayLike, y: ArrayLike, U: ArrayLike, xrange: tuple, yrange: tuple) -> tuple:
    """Crops the potential to the boundaries specified by xrange and yrange. 

    Args:
        x (ArrayLike): one dimensional array of x-points
        y (ArrayLike): one dimensional array of y-points
        U (ArrayLike): two dimensional array of the potential.
        xrange (tuple): tuple of two floats that indicate the min and max range for the x-coordinate.
        yrange (tuple): tuple of two floats that indicate the min and max range for the y-coordinate.

    Returns:
        tuple: cropped x array, cropped y array, cropped potential array.
    """
    xmin_idx, xmax_idx = find_nearest(x, xrange[0]), find_nearest(x, xrange[1])
    ymin_idx, ymax_idx = find_nearest(y, yrange[0]), find_nearest(y, yrange[1])

    return x[xmin_idx:xmax_idx], y[ymin_idx:ymax_idx], U[xmin_idx:xmax_idx, ymin_idx:ymax_idx]