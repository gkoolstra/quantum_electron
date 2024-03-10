import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Optional, List
import pyvista
from shapely import Polygon
import shapely.plotting
from matplotlib import pyplot as plt
from scipy.constants import elementary_charge as qe, epsilon_0
from scipy.constants import Boltzmann as kB
import matplotlib
import importlib

def package_versions():
    for module in ['quantum_electron', 'numpy', 'scipy', 'matplotlib']:
        globals()[module] = importlib.import_module(module)
        print(globals()[module].__name__, globals()[module].__version__)

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
    
def make_potential(potential_dict: Dict[str, ArrayLike], voltages: Dict[str, float]) -> ArrayLike:
    """Creates a numpy array potential based on an array of coupling coefficient arrays stored in potential_dict. 
    The returned potential values are positive for a positive voltage applied to the gate. Therefore, to transform
    the potential into potential energy, multiply with -1.

    Args:
        potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
        The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
        voltages (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
        applied to each electrode

    Returns:
        ArrayLike: Inner product of the coupling coefficient arrays and the voltages. 
    """

    for k, key in enumerate(list(voltages.keys())):
        if k == 0: 
            potential = potential_dict[key] * voltages[key] 
        else:
            potential += potential_dict[key] * voltages[key]
    
    return potential

def find_minimum_location(potential_dict: Dict[str, ArrayLike], voltages: Dict[str, float], return_potential_value: bool=False) -> tuple[float, float]:
    """Find the coordinates of the minimum energy point for a single electron.

    Args:
        potential_dict (Dict[str, ArrayLike]): Potential dictionary.
        voltages (Dict[str, float]): Voltage dictionary.
        return_potential_value (bool): Returns the value of the potential energy for a single electron at the minimum location.

    Returns:
        tuple[float, float]: (x_min, y_min, V_min) where the potential energy for a single electron is minimized. Units are in micron, eV.
    """
    
    potential = make_potential(potential_dict, voltages)
    zdata = -potential.T
    
    xidx, yidx = np.unravel_index(zdata.argmin(), zdata.shape)
    
    if return_potential_value:
        return potential_dict['xlist'][yidx], potential_dict['ylist'][xidx], zdata[xidx, yidx]
    else:
        return potential_dict['xlist'][yidx], potential_dict['ylist'][xidx]

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

class PotentialVisualization:
    def __init__(self, potential_dict: Dict[str, ArrayLike], voltages: Dict[str, float]):
        self.potential_dict = potential_dict
        self.voltage_dict = voltages    

    def plot_potential_energy(self, ax=None, coor: Optional[List[float]]=[0,0], dxdy: List[float]=[1, 2], figsize: tuple[float, float]=(7, 4), 
                              print_voltages: bool=True,  plot_contours: bool=True) -> None:
        """Plot the potential energy as function of (x,y)

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            figsize (tuple[float, float], optional): Figure size that gets passed to matplotlib.pyplot.figure. Defaults to (7, 4).
        """

        potential = make_potential(self.potential_dict, self.voltage_dict)
        zdata = -potential.T

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            make_colorbar = True
        else:
            make_colorbar = False
            
        pcm = ax.pcolormesh(self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, cmap=plt.cm.RdYlBu_r)
        
        if make_colorbar:
            cbar = plt.colorbar(pcm)
            tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.set_ylabel(r"Potential energy $-eV(x,y)$")
        
        xidx, yidx = np.unravel_index(zdata.argmin(), zdata.shape)
        ax.plot(self.potential_dict['xlist'][yidx], self.potential_dict['ylist'][xidx], '*', color='white')

        ax.set_xlim(coor[0] - dxdy[0]/2, coor[0] + dxdy[0]/2)
        ax.set_ylim(coor[1] - dxdy[1]/2, coor[1] + dxdy[1]/2)

        ax.set_aspect('equal')
        
        if print_voltages:
            for k, electrode in enumerate(self.voltage_dict.keys()):
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                ax.text(coor[0] - dxdy[0]/2 - 0.3 * (xmax - xmin), coor[1] + dxdy[1]/2 - k * 0.1 * (ymax - ymin), 
                        f"{electrode} = {self.voltage_dict[electrode]:.2f} V", ha='right', va='top')

        if plot_contours:
            contours = [np.round(np.min(zdata), 3) +k*1e-3 for k in range(5)]
            CS = ax.contour(self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, levels=contours)
            ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        ax.locator_params(axis='both', nbins=4)
        
        if ax is None:
            plt.tight_layout()