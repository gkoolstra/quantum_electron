import scipy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from .utils import find_nearest, xy2r, r2xy
from .schrodinger_solver import find_minimum_location, make_potential
from .position_solver import PositionSolver, ConvergenceMonitor
from .eom_solver import EOMSolver
from scipy.signal import convolve2d
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e

from numpy.typing import ArrayLike
from typing import List, Dict, Optional


class FullModel(EOMSolver, PositionSolver):
    def __init__(self, potential_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, float], 
                 f0: float = 4e9, Z0: float = 50, include_screening : bool = False, screening_length : float = np.inf, 
                 potential_smoothing: float = 5e-4, remove_unbound_electrons : bool = False, remove_bounds : Optional[tuple] = None, 
                 trap_annealing_steps: list = [0.1] * 5, max_x_displacement: float = 0.2e-6, max_y_displacement: float = 0.2e-6) -> None:
        """This class can be used to determine the coordinates of electrons in an electrostatic potential and solve for the in-plane equations of motion.
        Typical usage:
        
        voltage_dict = {"trap" : 0.5, "res_plus" : 0.4, "res_min" : 0.4}
        fm = FullModel(potential_dict, voltage_dict)
        fm.set_rf_interpolator(rf_electrode_labels=["res_plus", "res_minus"])
        fm.get_trap_electron_positions(n_electrons=5)

        Args:
            potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
        """
        self.rf_interpolator = None

        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict

        self.f0 = f0
        self.Z0 = Z0

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

        PositionSolver.__init__(self, potential_dict['xlist'] * 1e-6, potential_dict['ylist'] * 1e-6, -potential,
                                spline_order_x=self.spline_order, spline_order_y=self.spline_order,
                                smoothing=self.potential_smoothing, include_screening=self.include_screening, screening_length=self.screening_length)

        EOMSolver.__init__(self, self.f0, self.Z0, Ex=self.Ex, Ey=self.Ey,
                           curv_xx=self.ddVdx, curv_xy=self.ddVdxdy, curv_yy=self.ddVdy)

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

        assert len(rf_electrode_labels) >= 2
        rf_voltage_dict[rf_electrode_labels[0]] = +0.5
        rf_voltage_dict[rf_electrode_labels[1]] = -0.5

        potential = make_potential(self.potential_dict, rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator = scipy.interpolate.RectBivariateSpline(self.potential_dict['xlist']*1e-6,
                                                                     self.potential_dict['ylist']*1e-6,
                                                                     potential)

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

    def generate_initial_condition(self, n_electrons: int) -> ArrayLike:
        """Generates an initial condition for an arbitrary number of electrons. The coordinates are organized in a circular fashion and 
        are centered around the potential minimum.

        Args:
            n_electrons (int): Number of electrons.

        Returns:
            ArrayLike: One-dimensional array (length = 2 * n_electrons) of x and y coordinates: [x0, y0, x1, y0, ...]
        """

        coor = find_minimum_location(self.potential_dict, self.voltage_dict)

        # Generate initial guess positions for the electrons in a circle with certain radius.
        radius = 0.18E-6

        init_trap_x = np.array([coor[0] * 1e-6 + radius * np.cos(2 *
                               np.pi * n / float(n_electrons)) for n in range(n_electrons)])
        init_trap_y = np.array([coor[1] * 1e-6 + radius * np.sin(2 *
                               np.pi * n / float(n_electrons)) for n in range(n_electrons)])

        electron_initial_positions = xy2r(
            np.array(init_trap_x), np.array(init_trap_y))
        return electron_initial_positions

    def count_electrons_in_dot(self, r: ArrayLike, trap_bounds_x: tuple = (-0.9e-6, 0.8e-6), trap_bounds_y: tuple = (-0.9e-6, 0.9e-6)) -> float:
        """Counts the number of coordinate pairs in r that fall within the bounds specified by trap_bounds_x and trap_bounds_y

        Args:
            r (ArrayLike): Electron coordinates of length 2 * n_electrons. This should be in the order [x0, y0, x1, y1, ...]
            trap_bounds_x (tuple, optional): Electrons will be counted if they fall within this x-domain. The unit is meters. Defaults to (-0.9e-6, 0.8e-6).
            trap_bounds_y (tuple, optional): Electrons will be counted if they fall within this y-domain. The unit is meters. Defaults to (-0.9e-6, 0.9e-6).

        Returns:
            float: Number of electrons within the confines of the dot.
        """
        ex, ey = r2xy(r)
        x_ok = np.logical_and(ex < trap_bounds_x[1], ex > trap_bounds_x[0])
        y_ok = np.logical_and(ey < trap_bounds_y[1], ey > trap_bounds_y[0])
        x_and_y_ok = np.logical_and(x_ok, y_ok)
        return np.sum(x_and_y_ok)

    def get_trap_electron_positions(self, n_electrons: int, electron_initial_positions: Optional[ArrayLike] = None, verbose: bool = False,
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

        self.CM = self.ConvergenceMonitor(
            self.Vtotal, self.grad_total, call_every=1, verbose=verbose)

        # NOTE: Need to check about these numbers.
        gradient_tolerance = 1e-19
        epsilon = 1e-19

        trap_minimizer_options = {'method': 'L-BFGS-B',
                                  'jac': self.grad_total,
                                  'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                                  'callback': self.CM.monitor_convergence}

        # initial_jacobian = self.grad_total(electron_initial_positions)
        res = scipy.optimize.minimize(
            self.Vtotal, electron_initial_positions, **trap_minimizer_options)

        while res['status'] > 0:
            # Try removing unbounded electrons and restart the minimization
            if self.remove_unbound_electrons:
                # Remove any electrons that are to the left of the trap
                best_x, best_y = r2xy(res['x'])
                idxs = np.where(np.logical_and(
                    best_x > self.remove_bounds[0], best_x < self.remove_bounds[1]))[0]
                best_x = np.delete(best_x, idxs)
                best_y = np.delete(best_y, idxs)
                # Use the solution from the current time step as the initial condition for the next timestep!
                electron_initial_positions = xy2r(best_x, best_y)
                if len(best_x) < len(res['x'][::2]) and (not suppress_warnings):
                    print("%d/%d unbounded electrons removed. %d electrons remain." % (
                        int(len(res['x'][::2]) - len(best_x)), len(res['x'][::2]), len(best_x)))
                res = scipy.optimize.minimize(
                    self.Vtotal, electron_initial_positions, **trap_minimizer_options)
            else:
                best_x, best_y = r2xy(res['x'])
                idxs = np.union1d(np.where(best_x < -2E-6)
                                  [0], np.where(np.abs(best_y) > 2E-6)[0])
                if len(idxs) > 0 and (not suppress_warnings):
                    print("Following electrons are outside the simulation domain")
                    for i in idxs:
                        print("(x,y) = (%.3f, %.3f) um" %
                              (best_x[i] * 1E6, best_y[i] * 1E6))
                # To skip the infinite while loop.
                break

        if res['status'] > 0:
            if not suppress_warnings:
                cprint(
                    "WARNING: Initial minimization for Trap did not converge!", "red")
                print(
                    f"Final L-inf norm of gradient = {np.amax(res['jac']):.2f} eV/m")
                best_res = res
                cprint(
                    "Please check your initial condition, are all electrons confined in the simulation area?", "red")

        if len(self.trap_annealing_steps) > 0:
            if verbose:
                cprint("SUCCESS: Initial minimization for Trap converged!", "green")
                # This maps the electron positions within the simulation domain
                cprint("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) um..."
                       % (len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                          np.mean(self.thermal_kick_x(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dx=self.max_x_displacement)) * 1E6,
                          np.mean(self.thermal_kick_y(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                      maximum_dy=self.max_y_displacement)) * 1E6),
                       "white")

            best_res = self.perturb_and_solve(self.Vtotal, len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                                              res, maximum_dx=self.max_x_displacement, maximum_dy=self.max_y_displacement,
                                              do_print=verbose, 
                                              **trap_minimizer_options)
        else:
            best_res = res

        return best_res
