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


def select_outer_electrons(xi: ArrayLike, yi: ArrayLike, plot: bool = True, **kwargs) -> tuple:
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
        points = np.c_[xi.reshape(-1), yi.reshape(-1),
                       np.zeros(len(yi)).reshape(-1)]
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
    nearest_neighbor_distance = 1 / \
        np.sqrt(np.pi * density_from_positions(xi, yi))
    return qe ** 2 / (4 * np.pi * epsilon_0 * nearest_neighbor_distance) / (kB * T)


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
    idx = (np.abs(array-value)).argmin()
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


def find_minimum_location(potential_dict: Dict[str, ArrayLike], voltages: Dict[str, float], return_potential_value: bool = False) -> tuple[float, float]:
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
            
    def plot_coupling_constant_ratio(self, electrode1: str, electrode2: Optional[str], loc: tuple = (-1, 0), ax=None, coor: Optional[List[float]] = [0, 0], dxdy: List[float] = [1, 2], 
                                     figsize: tuple[float, float] = (7, 4), contour_levels: ArrayLike = [], clim: Optional[tuple] = None) -> float:
        """Plots and returns the ratio of coupling constants for a set of two electrodes, electrode1 and electrode2. electrode1 and electrode2 must be present as keys in 
        voltage_dict. loc controls the location at which the ratio of coupling constants is evaluated and returned.

        Args:
            electrode1 (str): Electrode name
            electrode2 (str): Electrode name, may be None. If None, only the coupling constant of electrode 1 is plotted.
            loc (tuple, optional): Location where the ratio electrode1/electrode2 is evaluated. Defaults to (-1, 0).
            ax (_type_, optional): Matplotlib axes instance. If None, a new instance will be created. Defaults to None.
            coor (Optional[List[float]], optional): Center for the 2D plot in units of microns. Defaults to [0, 0].
            dxdy (List[float], optional): Extent (dx, dy) of the 2D plot in units of microns. Defaults to [1, 2].
            figsize (tuple[float, float], optional): Matplotlib figure size in inches. Defaults to (7, 4).
            show_minimum (bool, optional): If True, it plots a star where the ratio is smallest. Defaults to True.
            contour_levels (ArrayLike, optional): Contour levels, must be a list. Defaults to [].
            clim (Optional[tuple], optional): Limits for the colorbar. Defaults to None.

        Returns:
            float: Ratio of the coupling constants evaluated at the location specified by 'loc'.
        """
        # Take the ratio between the two coupling constants
        if electrode2 is not None:
            zdata = self.potential_dict[electrode1].T / self.potential_dict[electrode2].T
        else:
            zdata = self.potential_dict[electrode1].T

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            make_colorbar = True
        else:
            make_colorbar = False

        if clim is not None:
            pcm = ax.pcolormesh(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, vmin=clim[0], vmax=clim[1], cmap=plt.cm.RdYlBu_r)
        else:
            pcm = ax.pcolormesh(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, cmap=plt.cm.RdYlBu_r)

        if make_colorbar:
            cbar = plt.colorbar(pcm, fraction=0.046, pad=0.04)
            tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.set_ylabel(f"Ratio of {electrode1} and {electrode2}")

        ax.set_xlim(coor[0] - dxdy[0]/2, coor[0] + dxdy[0]/2)
        ax.set_ylim(coor[1] - dxdy[1]/2, coor[1] + dxdy[1]/2)
        ax.set_aspect('equal')

        if len(contour_levels) >= 1:
            CS = ax.contour(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, levels=contour_levels)
            ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        ax.locator_params(axis='both', nbins=4)

        if ax is None:
            plt.tight_layout()
            
        xidx = np.argmin(np.abs(self.potential_dict['xlist'] - loc[0]))
        yidx = np.argmin(np.abs(self.potential_dict['ylist'] - loc[1]))
        ax.plot(loc[0], loc[1], '*', color='white')
        
        return zdata[yidx, xidx]
        
    def plot_coupling_constants(self, loc: tuple = (-1, 0), plot_coor: tuple = (0, 0), plot_dxdy: tuple = (3., 2.), clim: tuple = (0, 1)) -> dict:
        """Plots the grid of all coupling constants / lever arms in the potential dict. These are the potential maps when 1V is applied to each 
        respective electrode. It also returns the values of the coupling constants at the location loc = (x, y) in a dictionary.

        Args:
            loc (tuple, optional): Location to evaluate the coupling constants at (x, y) in micron. Defaults to (-1, 0).
            plot_coor (tuple, optional): Center for each of the 2D plots. Defaults to (0, 0).
            plot_dxdy (tuple, optional): Extent (width and height) in microns for each of the 2D plots. Defaults to (3., 2.).
            clim (tuple, optional): Colorbar limits for each of the 2D plots. Defaults to (0, 1).

        Returns:
            dict: Dictionary with electrode names as keys, and the value of the coupling constants evaluated at the position set by loc.
        """
        num_electrodes = len(self.voltage_dict.keys())
        num_rows = int(np.ceil(num_electrodes / 3))
        aspect = plot_dxdy[1] / plot_dxdy[0]
        
        initial_voltage_dict = self.voltage_dict.copy()

        fig, axs = plt.subplots(num_rows, 3, figsize=(3 * 3, num_rows * 3 * aspect))

        k = 0
        coupling_coeffs = {}
        for ax, electrode_name in zip(axs.flatten(), self.voltage_dict.keys()):
            for el in self.voltage_dict.keys():
                self.voltage_dict[el] = 1 if el == electrode_name else 0
                            
            # Note the colorbar limits are reversed, because the potential energy is -1 x coupling constants.
            self.plot_potential_energy(ax=ax, coor=plot_coor, dxdy=plot_dxdy, print_voltages=False, 
                                       plot_contours=False, show_minimum=False, clim=(-clim[1], -clim[0]))
            
            if k % 3 != 0:
                ax.set_ylabel("")
            if k < (num_electrodes - 3):
                ax.set_xlabel("")
            
            xidx = np.argmin(np.abs(self.potential_dict['xlist'] - loc[0]))
            yidx = np.argmin(np.abs(self.potential_dict['ylist'] - loc[1]))
            
            potential = make_potential(self.potential_dict, self.voltage_dict).T
            coupling_coeffs[electrode_name] = potential[yidx, xidx]
            
            ax.plot(loc[0], loc[1], '*', color='white')
            ax.set_title(f"{electrode_name}: {coupling_coeffs[electrode_name]:.3f}")
            
            k += 1
            
        # Don't show the remaining subplots in the grid.
        for jj in range(k, len(axs.flatten())):
            axs.flatten()[jj].axis("off")
            
        fig.tight_layout()
        
        # Reset the voltage dict:
        self.voltage_dict = initial_voltage_dict
        
        return coupling_coeffs
        
    def plot_potential_slice(self, ax=None, x: ArrayLike = [], y: ArrayLike = [], axlims: Optional[tuple] = None, 
                             figsize: tuple[float, float] = (6, 3), print_voltages: bool = True, 
                             tag: str = 'auto'):
        """Plot a potential slice along x or y. To control the dimension, supply arguments in one of the two forms
        - x = [x0], y = np.linspace(ymin, ymax, ...) to plot the potential vs. y at x = x0 OR
        - y = [y0], x = np.linspace(xmin, xmax, ...) to plot the potential vs. x at y = y0

        Args:
            ax (_type_, optional): Matplotlib axes object. If None, a new instance will be created. Defaults to None.
            x (ArrayLike, optional): x values for the potential slice. Must be at least of length 1. Defaults to [].
            y (ArrayLike, optional): y values for the potential slice. Must be at least of length 1. Defaults to [].
            axlims (Optional[tuple], optional): Limits in eV of the vertical axis of the plot. Defaults to None.
            figsize (tuple[float, float], optional): Figure size in inches. Defaults to (6, 3).
            print_voltages (bool, optional): Prints the voltages for each potential next to the plot. Defaults to True.
            tag (str, optional): Label in the legend that goes into the legend. If auto, the label is either x0 or y0. Defaults to 'auto'.

        Raises:
            ValueError: If x and y are not according to the rules above, a ValueError is raised.
        """
        potential = make_potential(self.potential_dict, self.voltage_dict)
        zdata = -potential.T
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
        if len(x) == 1: 
            # We are plotting along the y-axis for one particular value of x
            x_idx = np.argmin(np.abs(self.potential_dict['xlist'] - x[0]))
            label = rf"$x$ = {x[0]:.2f}"+f" {chr(956)}m" if tag == 'auto' else tag
            
            ax.plot(self.potential_dict['ylist'], zdata[:, x_idx], label=label)
            ax.set_xlabel("$y$"+f" ({chr(956)}m)")
            ax.set_ylabel(r"Potential energy $-eV(x,y)$")
            if axlims is not None:
                ax.set_ylim(axlims)
                
            ax.set_xlim(np.min(y), np.max(y))
            ax.locator_params(axis='both', nbins=4)
            ax.legend(loc=0, frameon=False)
            
            if print_voltages:
                for k, electrode in enumerate(self.voltage_dict.keys()):
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()

                    ax.text(xmin - 0.3 * (xmax - xmin), ymax - k * 0.1 * (ymax - ymin),
                            f"{electrode} = {self.voltage_dict[electrode]:.2f} V", ha='right', va='top')
            
        elif len(y) == 1: 
            # We are plotting along the x-axis for one particular value of y
            y_idx = np.argmin(np.abs(self.potential_dict['ylist'] - y[0]))
            label=rf"$y$ = {y[0]:.2f}"+f" {chr(956)}m" if tag == 'auto' else tag
            
            ax.plot(self.potential_dict['xlist'], zdata[y_idx, :], label=label)
            ax.set_xlabel("$x$"+f" ({chr(956)}m)")
            ax.set_ylabel(r"Potential energy $-eV(x,y)$")
            if axlims is not None:
                ax.set_ylim(axlims)
            ax.set_xlim(np.min(x), np.max(x))
            ax.locator_params(axis='both', nbins=4)
            ax.legend(loc=0, frameon=False)
            
            if print_voltages:
                for k, electrode in enumerate(self.voltage_dict.keys()):
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()

                    ax.text(xmin - 0.3 * (xmax - xmin), ymax - k * 0.1 * (ymax - ymin),
                            f"{electrode} = {self.voltage_dict[electrode]:.2f} V", ha='right', va='top')
            
        else:
            raise ValueError("At least one of 'x' or 'y' must contain only 1 element to indicate a slice along 'x' or 'y'.")
            
        
    def plot_potential_energy(self, ax=None, coor: Optional[List[float]] = [0, 0], dxdy: List[float] = [1, 2], figsize: tuple[float, float] = (7, 4),
                              show_minimum: bool = True, print_voltages: bool = True,  plot_contours: bool = True, clim: Optional[tuple] = None) -> None:
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

        if clim is not None:
            pcm = ax.pcolormesh(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, vmin=clim[0], vmax=clim[1], cmap=plt.cm.RdYlBu_r)
        else:
            pcm = ax.pcolormesh(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, cmap=plt.cm.RdYlBu_r)

        if make_colorbar:
            cbar = plt.colorbar(pcm, fraction=0.046, pad=0.04)
            tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.set_ylabel(r"Potential energy $-eV(x,y)$")

        xidx, yidx = np.unravel_index(zdata.argmin(), zdata.shape)
        ax.plot(self.potential_dict['xlist'][yidx],
                self.potential_dict['ylist'][xidx], '*', color='white')

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
            contours = [np.round(np.min(zdata), 3) + k*1e-3 for k in range(5)]
            CS = ax.contour(
                self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, levels=contours)
            ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        ax.locator_params(axis='both', nbins=4)

        if ax is None:
            plt.tight_layout()
