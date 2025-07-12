import scipy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
import matplotlib.animation as animation
import shapely
from shapely import Polygon
import numpy as np
from .utils import find_nearest, xy2r, r2xy, find_minimum_location, make_potential
from .utils import PotentialVisualization
from .position_solver import PositionSolver, ConvergenceMonitor
from .eom_solver import EOMSolver
from scipy.signal import convolve2d
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e
from skimage import measure
from scipy.interpolate import interp1d

from numpy.typing import ArrayLike
from typing import List, Dict, Optional


class FullModel(EOMSolver, PositionSolver, PotentialVisualization):
    def __init__(self, potential_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, float],
                 include_screening: bool = False, screening_length: float = np.inf,
                 potential_smoothing: float = 5e-4, remove_unbound_electrons: bool = False, remove_bounds: Optional[tuple] = None,
                 trap_annealing_steps: list = [0.1] * 5, max_x_displacement: float = 0.2e-6, max_y_displacement: float = 0.2e-6) -> None:
        """This class can be used to determine the coordinates of electrons in an electrostatic potential and solve for the in-plane equations of motion.
        Typical usage:

        voltage_dict = {"trap" : 0.5, "res_plus" : 0.4, "res_min" : 0.4}
        fm = FullModel(potential_dict, voltage_dict)
        fm.set_rf_interpolator(rf_electrode_labels=["res_plus", "res_minus"])
        fm.get_electron_positions(n_electrons=5)

        Args:
            potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
        """
        self.rf_interpolator = None

        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict

        self.include_screening = include_screening
        self.screening_length = screening_length
        self.potential_smoothing = potential_smoothing
        self.spline_order = 3
        if remove_bounds is None:
            self.remove_bounds = (
                0.95*potential_dict['xlist'][0] * 1e-6, 0.95*potential_dict['xlist'][-1] * 1e-6)
        else:
            self.remove_bounds = remove_bounds
        self.remove_unbound_electrons = remove_unbound_electrons

        self.trap_annealing_steps = trap_annealing_steps
        self.max_x_displacement = max_x_displacement
        self.max_y_displacement = max_y_displacement

        potential = make_potential(potential_dict, voltage_dict)

        # Inherit methods from the PositionSolver, EOMSolver and PotentialVisualization classes
        PositionSolver.__init__(self, potential_dict['xlist'] * 1e-6, potential_dict['ylist'] * 1e-6, -potential,
                                spline_order_x=self.spline_order, spline_order_y=self.spline_order,
                                smoothing=self.potential_smoothing, include_screening=self.include_screening, screening_length=self.screening_length)

        EOMSolver.__init__(self, Ex=self.Ex, Ey=self.Ey,
                           Ex_up=self.Ex_up, Ex_down=self.Ex_down, Ey_up=self.Ey_up, Ey_down=self.Ey_down,
                           curv_xx=self.ddVdx, curv_xy=self.ddVdxdy, curv_yy=self.ddVdy)

        PotentialVisualization.__init__(
            self, potential_dict=potential_dict, voltages=voltage_dict)

        self.ConvergenceMonitor = ConvergenceMonitor

    def set_rf_interpolator(self, rf_electrode_labels: List[str]) -> None:
        """Sets the rf_interpolator object, which allows evaluation of the electric field Ex and Ey at arbitrary coordinates. 
        This must be done before any calls to EOMSolver, such as setup_eom or solve_eom.

        The RF field Ex and Ey are determined from the same data as the DC fields, and are evaluated by setting +/- 0.5V on the 
        electrodes that couple to the RF-mode. These electrodes should be specified in the argument rf_electrode_labels.

        Args:
            rf_electrode_labels (List[str]): List of electrode names, these strings must also be present as keys in voltage_dict and potential_dict.
        """

        rf_voltage_dict = self.voltage_dict.copy()

        # For generating the rf electrodes, we must set all electrodes to 0.0 except the resoantor + and resonator - electrodes.
        for key in rf_voltage_dict.keys():
            rf_voltage_dict[key] = 0.0

        if len(rf_electrode_labels) == 2:
            rf_voltage_dict[rf_electrode_labels[0]] = +0.5
            rf_voltage_dict[rf_electrode_labels[1]] = -0.5
        elif len(rf_electrode_labels) == 1:
            rf_voltage_dict[rf_electrode_labels[0]] = +1.0
        else:
            raise ValueError(
                "More than 2 electrodes are not supported for the RF interpolator.")

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator = scipy.interpolate.RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                                     self.potential_dict['ylist']*1e-6,
                                                                     potential)

        # The code below is only for setting up the coupled LC circuit.
        # For the coupled LC circuit, we must consider the electric field generated by each electrode individually
        # In this case, rf_electrode_labels must contain at least 2 items
        if len(rf_electrode_labels) == 1:
            rf_electrode_labels *= 2

        assert len(rf_electrode_labels) == 2

        # We assume the first electrode is associated with the 'up' electrode
        rf_voltage_dict[rf_electrode_labels[0]] = 1.0
        rf_voltage_dict[rf_electrode_labels[1]] = 0.0

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator_up = scipy.interpolate.RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                                        self.potential_dict['ylist']*1e-6,
                                                                        potential)

        # Repeat for the 'down' electrode
        rf_voltage_dict[rf_electrode_labels[0]] = 0.0
        rf_voltage_dict[rf_electrode_labels[1]] = 1.0

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator_down = scipy.interpolate.RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                                          self.potential_dict['ylist']*1e-6,
                                                                          potential)

    def Ex_up(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction due to only the `up` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_up.ev(xe, ye, dx=1)

    def Ex_down(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction due to only the `down` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_down.ev(xe, ye, dx=1)

    def Ey_up(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction due to only the `up` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_up.ev(xe, ye, dy=1)

    def Ey_down(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction due to only the `down` electrode in the differential pair. 
        `setup_rf_interpolator` must be run prior to calling this function.
        This function is used by the setup_eom_coupled_lc function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator_down.ev(xe, ye, dy=1)

    def Ex(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the x-direction from the rf_interpolator. setup_rf_interpolator must be run prior to calling this function.
        This function is used by the setup_eom function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ex should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ex should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator.ev(xe, ye, dx=1)

    def Ey(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction from the rf_interpolator. setup_rf_interpolator must be run prior to calling this function.
        This function is used by the setup_eom function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ey should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ey should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator.ev(xe, ye, dy=1)

    def generate_initial_condition(self, n_electrons: int, radius: float = 0.18E-6, center=None) -> ArrayLike:
        """Generates an initial condition for an arbitrary number of electrons. The coordinates are organized in a circular fashion and 
        are centered around the potential minimum.

        Args:
            n_electrons (int): Number of electrons.

        Returns:
            ArrayLike: One-dimensional array (length = 2 * n_electrons) of x and y coordinates: [x0, y0, x1, y0, ...]
        """
        if center is None:
            coor = find_minimum_location(
                self.potential_dict, self.voltage_dict)
        else:
            coor = center

        # Generate initial guess positions for the electrons in a circle with certain radius.
        init_trap_x = np.array([coor[0] * 1e-6 + radius * np.cos(2 *
                               np.pi * n / float(n_electrons)) for n in range(n_electrons)])
        init_trap_y = np.array([coor[1] * 1e-6 + radius * np.sin(2 *
                               np.pi * n / float(n_electrons)) for n in range(n_electrons)])

        electron_initial_positions = xy2r(
            np.array(init_trap_x), np.array(init_trap_y))
        return electron_initial_positions

    def count_electrons_in_dot(self, r: ArrayLike, trap_bounds_x: tuple = (-1e-6, 1e-6), trap_bounds_y: tuple = (-1e-6, 1e-6)) -> float:
        """Counts the number of coordinate pairs in r that fall within the bounds specified by trap_bounds_x and trap_bounds_y

        Args:
            r (ArrayLike): Electron coordinates of length 2 * n_electrons. This should be in the order [x0, y0, x1, y1, ...]
            trap_bounds_x (tuple, optional): Electrons will be counted if they fall within this x-domain. The unit is meters. Defaults to (-1e-6, 1e-6).
            trap_bounds_y (tuple, optional): Electrons will be counted if they fall within this y-domain. The unit is meters. Defaults to (-1e-6, 1e-6).

        Returns:
            float: Number of electrons within the confines of the dot.
        """
        ex, ey = r2xy(r)
        x_ok = np.logical_and(ex < trap_bounds_x[1], ex > trap_bounds_x[0])
        y_ok = np.logical_and(ey < trap_bounds_y[1], ey > trap_bounds_y[0])
        x_and_y_ok = np.logical_and(x_ok, y_ok)
        return np.sum(x_and_y_ok)

    def get_dot_area(self, plot: bool = True, barrier_location: tuple = (-1, 0), barrier_offset: float = -0.01, **kwargs) -> float:
        """Finds the area of the dot spanned by the points that lie on a equipotential that is determined by the 
        `barrier_location` and `barrier_offset`. The resulting area has the same units as self.potential_dict['xlist'] ** 2

        Args:
            plot (bool, optional): Plot the contour and polygon spanned by that contour. Defaults to True.
            barrier_location (tuple, optional): Location (x, y) in the map where to measure the barrier_height. The contour
            will be drawn `barrier_offset` above the potential value at the barrier_location. Defaults to (-1, 0).
            barrier_offset (float, optional): barrier_offset in eV. The contour will be drawn with this offset. Defaults to -0.01 (eV).

        Returns:
            float: Area
        """
        potential = make_potential(self.potential_dict, self.voltage_dict)

        idx = find_nearest(self.potential_dict['ylist'], barrier_location[1])
        idy = find_nearest(self.potential_dict['xlist'], barrier_location[0])
        barrier_height = -potential[idy, idx]

        # Contour can return non-integer indices (it interpolates to find the contour)
        # Thus we need to create a mappable for x and y.
        fx = interp1d(
            np.arange(len(self.potential_dict['xlist'])), self.potential_dict['xlist'])
        fy = interp1d(
            np.arange(len(self.potential_dict['ylist'])), self.potential_dict['ylist'])

        # Use sci-kit image function measure to find the contours.
        contours = measure.find_contours(-potential.T,
                                         barrier_height + barrier_offset)

        # There may be multiple contours, but hopefully just one.
        if len(contours) > 0:
            for contour in contours:
                xs = fx(contour[:, 1])
                ys = fy(contour[:, 0])

            p = Polygon(np.c_[xs, ys])

            if plot:
                shapely.plotting.plot_polygon(p, **kwargs)
                plt.grid(None)

            return p.area
        else:
            # If there are no contours, the situation is easy
            return 0.0

    def get_electron_positions(self, n_electrons: int, electron_initial_positions: Optional[ArrayLike] = None, verbose: bool = False,
                               suppress_warnings: bool = False) -> dict:
        """This is the main method to calculate the electron positions in an electrostatic potential. This function can be called with a specific initial condition, 
        which can be useful during voltage sweeps, or with the default initial condition as specified in generate_initial_condition.

        Upon running this function, useful feedback about the convergence can be found in the attribute CM

        Args:
            n_electrons (int): Number of electrons.
            electron_initial_positions (Optional[ArrayLike], optional): Electron initial positions in the form [x0, y0, x1, y1, ...]. Defaults to None.
            verbose (bool, optional): Prints convergence information. Defaults to False.
            suppress_warnings (bool, optional): If false, this prints warnings if the minimization fails to converge. Defaults to False.

        Returns:
            dict: Dictionary object returned from scipy.optimize.minimize. Some useful attributes in this dictionary: 'status' > 0 means the minimization failed. 
            'x' contains the best solution that minimizes the gradient contained in 'jac'.
        """

        if electron_initial_positions is None:
            electron_initial_positions = self.generate_initial_condition(
                n_electrons)

        if (len(electron_initial_positions) // 2 != n_electrons) and (not suppress_warnings):
            print(
                "WARNING: The initial condition does not match n_electrons. n_electrons is ignored.")

        self.CM = self.ConvergenceMonitor(
            self.Vtotal, self.grad_total, call_every=1, verbose=verbose)

        # Convergence can happen one of two ways
        # (a) if the gradient self.grad_total(res['x']) < gradient_tolerance
        # (b) if res['fun'] changes less than the floating point precision from one iteration to the next.
        gradient_tolerance = 1e-1  # Units are eV/m

        # For improved performance we use maxls=100. Default is 20, but if starting close to the final solution, sometimes more
        # line searches are needed to converge. This is also helpful if the function landscape is very flat.
        trap_minimizer_options = {'method': 'L-BFGS-B',
                                  'jac': self.grad_total,
                                  'options': {'disp': False, 'gtol': gradient_tolerance, 'maxls': 100},
                                  'callback': self.CM.monitor_convergence}

        # initial_jacobian = self.grad_total(electron_initial_positions)
        res = scipy.optimize.minimize(
            self.Vtotal, electron_initial_positions, **trap_minimizer_options)

        while res['status'] > 0:
            no_electrons_left = False

            # Try removing unbounded electrons and restart the minimization
            if self.remove_unbound_electrons:
                # Remove any electrons that are to the left of the trap
                best_x, best_y = r2xy(res['x'])
                idcs_x = np.where(np.logical_or(
                    best_x < self.remove_bounds[0], best_x > self.remove_bounds[1]))[0]
                idcs_y = np.where(np.logical_or(
                    best_y < self.remove_bounds[0], best_y > self.remove_bounds[1]))[0]

                all_idcs_to_remove = np.union1d(idcs_x, idcs_y)
                best_x = np.delete(best_x, all_idcs_to_remove)
                best_y = np.delete(best_y, all_idcs_to_remove)

                # Use the solution from the current time step as the initial condition for the next timestep!
                electron_initial_positions = xy2r(best_x, best_y)
                if len(best_x) < len(res['x'][::2]) and (not suppress_warnings):
                    print("%d/%d unbounded electrons removed. %d electrons remain." % (
                        int(len(res['x'][::2]) - len(best_x)), len(res['x'][::2]), len(best_x)))
                else:  # sometimes the simulation doesn't converge for other reasons...
                    break

                if len(electron_initial_positions) > 0:
                    print("Restart minimization!")
                    self.CM = self.ConvergenceMonitor(
                        self.Vtotal, self.grad_total, call_every=1, verbose=verbose)
                    trap_minimizer_options['callback'] = self.CM.monitor_convergence
                    res = scipy.optimize.minimize(
                        self.Vtotal, electron_initial_positions, **trap_minimizer_options)
                else:
                    no_electrons_left = True
                    break
            else:
                best_x, best_y = r2xy(res['x'])
                idxs = np.union1d(np.where(best_x < self.x_min)
                                  [0], np.where(np.abs(best_y) > self.x_max)[0])
                if len(idxs) > 0 and (not suppress_warnings):
                    print("Following electrons are outside the simulation domain")
                    for i in idxs:
                        print("(x,y) = (%.3f, %.3f) um" %
                              (best_x[i] * 1E6, best_y[i] * 1E6))
                # To skip the infinite while loop.
                break

        if res['status'] > 0 and not (no_electrons_left) and not (suppress_warnings):
            print("WARNING: Initial minimization did not converge!")
            print(
                f"Final L-inf norm of gradient = {np.amax(res['jac']):.2f} eV/m")
            best_res = res
            print(
                "Please check your initial condition, are all electrons confined in the simulation area?")

        if len(self.trap_annealing_steps) > 0:
            if verbose:
                print("SUCCESS: Initial minimization for Trap converged!")
                # This maps the electron positions within the simulation domain
                print("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) µm..."
                      % (len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                          np.mean(self.thermal_kick_x(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dx=self.max_x_displacement)) * 1E6,
                          np.mean(self.thermal_kick_y(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dy=self.max_y_displacement)) * 1E6))

            best_res = self.perturb_and_solve(self.Vtotal, len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                                              res, maximum_dx=self.max_x_displacement, maximum_dy=self.max_y_displacement,
                                              do_print=verbose,
                                              **trap_minimizer_options)
        else:
            best_res = res

        if self.remove_unbound_electrons:
            best_x, best_y = r2xy(best_res['x'])
            idcs_x = np.where(np.logical_or(best_x < self.remove_bounds[0],
                                            best_x > self.remove_bounds[1]))[0]
            idcs_y = np.where(np.logical_or(best_y < self.remove_bounds[0],
                                            best_y > self.remove_bounds[1]))[0]

            all_idcs_to_remove = np.union1d(idcs_x, idcs_y)
            best_x = np.delete(best_x, all_idcs_to_remove)
            best_y = np.delete(best_y, all_idcs_to_remove)

            best_res['x'] = xy2r(best_x, best_y)

        return best_res

    def plot_electron_positions(self, res: dict, ax=None, color: str = 'mediumseagreen', marker_size: float = 10.0, shadow: bool=True, **kwargs) -> None:
        """Plot electron positions obtained from get_electron_positions

        Args:
            res (dict): Results dictionary from scipy.optimize.minimize
            ax (_type_, optional): Matplotlib axes object. Defaults to None.
            color (str, optional): Color of the markers representing the electrons. Defaults to 'mediumseagreen'.
        """
        x, y = r2xy(res['x'])

        if ax is None:
            if shadow:
                plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                        path_effects=[pe.SimplePatchShadow(), pe.Normal()], **kwargs)
            else:
                plt.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size, **kwargs)
        else:
            if shadow:
                ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size,
                        path_effects=[pe.SimplePatchShadow(), pe.Normal()], **kwargs)
            else:
                ax.plot(x*1e6, y*1e6, 'ok', mfc=color, mew=0.5, ms=marker_size, **kwargs)

    def animate_voltage_sweep(self, fig, ax, list_of_voltages: list, list_of_electron_positions: list, coor: tuple = (0, 0), dxdy: tuple = (2, 2), 
                              frame_interval_ms: int = 10, print_voltages: bool = False) -> matplotlib.animation.FuncAnimation:
        """
        Animates a voltage sweep by updating the voltage and electron positions over time. 
        This function only animates the sweep, it does not calculate the electron positions. This needs to be done beforehand.

        Args:
            list_of_voltages (list): A list of dictionaries representing the voltages at each frame.
            list_of_electron_positions (list): A list of arrays representing the electron positions at each frame.
            coor (tuple, optional): The coordinates of the center of the plot. Defaults to (0, 0).
            dxdy (tuple, optional): The width and height of the plot. Defaults to (2, 2).
            frame_interval_ms (int, optional): The time interval between frames in milliseconds. Defaults to 10.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.

        Raises:
            AssertionError: If the length of the voltage list is not the same as the list of electron positions.
        """
        assert len(list_of_voltages) == len(
            list_of_electron_positions), "The length of the voltage list must be the same as the list of electron positions."

        potential = make_potential(self.potential_dict, list_of_voltages[0])
        zdata = -potential.T

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        img_data = ax.imshow(zdata[::-1, :], cmap=plt.cm.RdYlBu_r, extent=[np.min(self.potential_dict['xlist']), np.max(self.potential_dict['xlist']),
                                                                           np.min(self.potential_dict['ylist']), np.max(self.potential_dict['ylist'])])

        final_x, final_y = r2xy(list_of_electron_positions[0])
        pts_data = ax.plot(final_x*1e6, final_y*1e6, 'ok', mfc='mediumseagreen', mew=0.5, ms=10,
                           path_effects=[pe.SimplePatchShadow(), pe.Normal()])

        cbar = plt.colorbar(img_data, fraction=0.046, pad=0.04)
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.set_ylabel(r"Potential energy $-eV(x,y)$")

        xmin, xmax = (coor[0] - dxdy[0]/2, coor[0] + dxdy[0]/2)
        ymin, ymax = (coor[1] - dxdy[1]/2, coor[1] + dxdy[1]/2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        text_boxes = list()
        initial_voltages = list_of_voltages[0]
        if print_voltages:
            for k, electrode in enumerate(initial_voltages.keys()):
                text_boxes.append(ax.text(xmin - 0.75,
                                        ymax - k * 0.075 * (ymax - ymin),
                                        f"{electrode} = {initial_voltages[electrode]:.2f} V", ha='right', va='top'))

        ax.set_aspect('equal')
        ax.set_xlabel("$x$"+f" ({chr(956)}m)")
        ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        plt.locator_params(axis='both', nbins=4)

        fig.tight_layout()

        def update(frame):
            # Update the voltages and electron positions
            voltages = list_of_voltages[frame]
            final_x, final_y = r2xy(list_of_electron_positions[frame])

            potential = make_potential(self.potential_dict, voltages)
            zdata = -potential.T

            # Update the color plot
            img_data.set_data(zdata[::-1, :])

            # Update the electron positions (green dots)
            pts_data[0].set_xdata(final_x * 1e6)
            pts_data[0].set_ydata(final_y * 1e6)

            if print_voltages:
                # Update the voltages to the left of the image
                for k, electrode in enumerate(voltages.keys()):
                    text_boxes[k].set_text(
                        f"{electrode} = {voltages[electrode]:.2f} V")

            return (img_data, pts_data, text_boxes)

        return animation.FuncAnimation(fig=fig, func=update, frames=np.arange(len(list_of_voltages)), interval=frame_interval_ms, repeat=True)

    def animate_convergence(self, fig, ax, coor: tuple = (0, 0), dxdy: tuple = (2, 2), frame_interval_ms: int = 10) -> matplotlib.animation.FuncAnimation:
        """Animate the convergence data stored in the convergence helper class. 

        Args:
            coor (tuple, optional): The coordinates of the center of the plot. Defaults to (0, 0).
            dxdy (tuple, optional): The width and height of the plot. Defaults to (2, 2).
            frame_interval_ms (int, optional): Interval between frames in milliseconds. Defaults to 10.

        Returns:
            _type_: matplotlib.animation.FuncAnimation object.
        """
        # The position data is stored in the coordinates of the helper class
        r = self.CM.curr_xk

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        self.plot_potential_energy(
            ax=ax, coor=coor, dxdy=dxdy, print_voltages=False, plot_contours=False)

        rx, ry = r2xy(r[0, :])
        pts_data = ax.plot(rx*1e6, ry*1e6, 'ok', mfc='mediumseagreen', mew=0.5,
                           ms=10, path_effects=[pe.SimplePatchShadow(), pe.Normal()])

        # Only things in the update function will get updated.
        def update(frame):
            rx, ry = r2xy(r[frame, :])
            # Update the electron positions (green dots)
            pts_data[0].set_xdata(rx * 1e6)
            pts_data[0].set_ydata(ry * 1e6)

            return pts_data,

        fig.tight_layout()
        # The interval is in milliseconds
        return animation.FuncAnimation(fig=fig, func=update, frames=np.arange(self.CM.curr_xk.shape[0]), interval=frame_interval_ms, repeat=True)

    def plot_convergence(self, ax=None) -> None:
        """Plot the convergence of the latest solution from get_electron_positions

        Args:
            ax (optional): Matplotlib axes object. Defaults to None.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5., 3.5))
        ax.plot(self.CM.curr_grad_norm)
        ax.set_yscale('log')
        ax.set_xlim(-1, len(self.CM.curr_grad_norm) + 1)
        ax.locator_params(axis='x', nbins=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost function")
