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
    def __init__(self, potential_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, float]):
        self.rf_interpolator = None

        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict 
        
        self.f0 = 4e9
        self.Z0 = 50
        
        self.include_screening = False
        self.screening_length = np.inf
        self.potential_smoothing = 1e-4
        self.spline_order = 3 
        self.remove_bounds = (0.95*potential_dict['xlist'][0] * 1e-6, 0.95*potential_dict['xlist'][-1] * 1e-6)
        self.remove_unbound_electrons = True
        
        self.trap_annealing_steps = [0.1] * 5
        self.max_x_displacement = 0.2e-6
        self.max_y_displacement = 0.2e-6
        
        potential = make_potential(potential_dict, voltage_dict)
        
        PositionSolver.__init__(self, potential_dict['xlist'] * 1e-6, potential_dict['ylist'] * 1e-6, -potential, 
                                spline_order_x=self.spline_order, spline_order_y=self.spline_order, 
                                smoothing=self.potential_smoothing, include_screening=self.include_screening, screening_length=self.screening_length)
        
        EOMSolver.__init__(self, self.f0, self.Z0, Ex=self.Ex, Ey=self.Ey, curv_xx=self.ddVdx, curv_xy=self.ddVdxdy, curv_yy=self.ddVdy)
        
        self.ConvergenceMonitor = ConvergenceMonitor

    def set_rf_interpolator(self, rf_electrode_labels: List[str]) -> None: 
        
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
        
    def Ex(self, xe, ye):
        return self.rf_interpolator.ev(xe, ye, dx=1)

    def Ey(self, xe, ye):
        return self.rf_interpolator.ev(xe, ye, dy=1)

    def crop_potential(self, x, y, U, xrange, yrange):
        xmin_idx, xmax_idx = find_nearest(x, xrange[0]), find_nearest(x, xrange[1])
        ymin_idx, ymax_idx = find_nearest(y, yrange[0]), find_nearest(y, yrange[1])

        return x[xmin_idx:xmax_idx], y[ymin_idx:ymax_idx], U[ymin_idx:ymax_idx, xmin_idx:xmax_idx]

    def generate_initial_condition(self, N):
        
        coor = find_minimum_location(self.potential_dict, self.voltage_dict)
        
        # Generate initial guess positions for the electrons in a circle with certain radius.
        radius = 0.18E-6
        
        init_trap_x = np.array([coor[0] * 1e-6 + radius * np.cos(2 * np.pi * n / float(N)) for n in range(N)])
        init_trap_y = np.array([coor[1] * 1e-6 + radius * np.sin(2 * np.pi * n / float(N)) for n in range(N)]) 

        electron_initial_positions = xy2r(np.array(init_trap_x), np.array(init_trap_y))
        return electron_initial_positions
    
    def count_electrons_in_dot(self, r, trap_bounds_x=(-0.9e-6, 0.8e-6), trap_bounds_y=(-0.9e-6, 0.9e-6)):
        ex, ey = r2xy(r)
        x_ok = np.logical_and(ex < trap_bounds_x[1], ex > trap_bounds_x[0])
        y_ok = np.logical_and(ey < trap_bounds_y[1], ey > trap_bounds_y[0])
        x_and_y_ok = np.logical_and(x_ok, y_ok)
        return np.sum(x_and_y_ok)

    def get_trap_electron_positions(self, n_electrons, electron_initial_positions: Optional[ArrayLike]=None, verbose: bool=False):

        if electron_initial_positions is None:
            electron_initial_positions = self.generate_initial_condition(n_electrons)
        
        self.CM = self.ConvergenceMonitor(self.Vtotal, self.grad_total, call_every=1, verbose=verbose)

        gradient_tolerance = 1e-19
        epsilon = 1e-19
        
        trap_minimizer_options = {'method': 'L-BFGS-B',
                                  'jac': self.grad_total,
                                  'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                                  'callback': self.CM.monitor_convergence}

        # initial_jacobian = self.grad_total(electron_initial_positions)
        res = scipy.optimize.minimize(self.Vtotal, electron_initial_positions, **trap_minimizer_options)
        
        while res['status'] > 0:
            # Try removing unbounded electrons and restart the minimization
            if self.remove_unbound_electrons:
                # Remove any electrons that are to the left of the trap
                best_x, best_y = r2xy(res['x'])
                idxs = np.where(np.logical_and(best_x > self.remove_bounds[0], best_x < self.remove_bounds[1]))[0]
                best_x = np.delete(best_x, idxs)
                best_y = np.delete(best_y, idxs)
                # Use the solution from the current time step as the initial condition for the next timestep!
                electron_initial_positions = xy2r(best_x, best_y)
                if len(best_x) < len(res['x'][::2]):
                    print("%d/%d unbounded electrons removed. %d electrons remain." % (
                    int(len(res['x'][::2]) - len(best_x)), len(res['x'][::2]), len(best_x)))
                res = scipy.optimize.minimize(self.Vtotal, electron_initial_positions, **trap_minimizer_options)
            else:
                best_x, best_y = r2xy(res['x'])
                idxs = np.union1d(np.where(best_x < -2E-6)[0], np.where(np.abs(best_y) > 2E-6)[0])
                if len(idxs) > 0:
                    print("Following electrons are outside the simulation domain")
                    for i in idxs:
                        print("(x,y) = (%.3f, %.3f) um" % (best_x[i] * 1E6, best_y[i] * 1E6))
                # To skip the infinite while loop.
                break

        if res['status'] > 0:
            cprint("WARNING: Initial minimization for Trap did not converge!", "red")
            print(f"Final L-inf norm of gradient = {np.amax(res['jac']):.2f} eV/m")
            best_res = res
            cprint("Please check your initial condition, are all electrons confined in the simulation area?", "red")

        if len(self.trap_annealing_steps) > 0:
            # cprint("SUCCESS: Initial minimization for Trap converged!", "green")
            # This maps the electron positions within the simulation domain
            cprint("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) um..." \
                    % (len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                        np.mean(self.thermal_kick_x(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                    maximum_dx=self.max_x_displacement)) * 1E6,
                        np.mean(self.thermal_kick_y(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                    maximum_dy=self.max_y_displacement)) * 1E6),
                    "white")
            best_res = self.perturb_and_solve(self.Vtotal, len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                                                res, maximum_dx=self.max_x_displacement, maximum_dy=self.max_y_displacement,
                                                **trap_minimizer_options)
        else:
            best_res = res
            
        return best_res