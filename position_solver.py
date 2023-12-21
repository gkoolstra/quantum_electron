import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import approx_fprime, minimize
from scipy.interpolate import RectBivariateSpline
import os, time, multiprocessing
from termcolor import cprint
from .utils import xy2r, r2xy
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e, Boltzmann as kB
from typing import Optional
from numpy.typing import ArrayLike

class ConvergenceMonitor:
    def __init__(self, Uopt: callable, grad_Uopt: callable, call_every: int, Uext: Optional[callable]=None, 
                 xext: Optional[ArrayLike]=None, yext: Optional[ArrayLike]=None, verbose: bool=True, eps: float=1E-12, save_path: Optional[str]=None,
                 figsize: tuple=(6.5,3.), coordinate_transformation: Optional[callable]=None, clim: tuple=(-0.75, 0)) -> None:
        """
        To be used with scipy.optimize.minimize as a call back function. One has two choices for call-back functions:
        - monitor_convergence: print the status of convergence (value of Uopt and norm of grad_Uopt)
        - save_pictures: save figures of the electron positions every iteration to construct a movie.
        :param Uopt: Cost function or total energy of the system. Uopt takes one argument and returns a scalar.
        :param grad_Uopt: Gradient of Uopt. This should be a function that takes one argument and returns
                          an array of size 2*N_electrons
        :param call_every: Report the status of optimization every N times. This should be an integer
        :param Uext: Electrostatic potential function. Takes 2 arguments (x,y) and returns an array of the size of x
        and y
        :param xext: Array of arbitrary size for evaluating the electrostatic potential. Units should be meters.
        :param yext: Array of arbitrary size for evaluating the electrostatic potential. Units should be meters.
        :param verbose: Whether to print the status when monitor_convergence is called.
        :param eps: Step size used to numerically approximate the gradient with scipy.optimize.approx_fprime
        :param save_path: Directory in which to save figures when self.save_pictures is called. None by default.
        """
        self.clim = clim
        self.call_every = call_every
        self.call_counter = 0
        self.verbose = verbose
        self.curr_grad_norm = list()
        self.curr_fun = list()
        self.iter = list()
        self.epsilon = eps
        self.save_path = save_path
        self.Uopt = Uopt
        self.grad_Uopt = grad_Uopt
        self.xext, self.yext, self.Uext = xext, yext, Uext
        self.figsize = figsize
        self.coordinate_transformation = coordinate_transformation
        self.electrode_outline_filename = None

    def monitor_convergence(self, xk: ArrayLike) -> None:
        """
        Monitor the convergence while the optimization is running. To be used with scipy.optimize.minimize.
        :param xk: Electron position pairs
        :return: None
        """
        if not (self.call_counter % self.call_every):
            self.iter.append(self.call_counter)
            self.curr_fun.append(self.Uopt(xk))

            # Here we use the L-inf norm (the maximum)
            self.curr_grad_norm.append(np.max(np.abs(self.grad_Uopt(xk))))

            if self.call_counter == 0:
                self.curr_xk = xk
                self.jac = self.grad_Uopt(xk)
                #self.approx_fprime = approx_fprime(xk, self.Uopt, self.epsilon)
            else:
                self.curr_xk = np.vstack((self.curr_xk, xk))
                self.jac = np.vstack((self.jac, self.grad_Uopt(xk)))
                #self.approx_fprime = np.vstack((self.approx_fprime, approx_fprime(xk, self.Uopt, self.epsilon)))

            if self.verbose:
                print("%d\tUopt: %.8f eV\tNorm of gradient: %.2e eV/m" \
                      % (self.call_counter, self.curr_fun[-1], self.curr_grad_norm[-1]))

        self.call_counter += 1

    def save_pictures(self, xk: ArrayLike) -> None:
        """
        Plots the current value of the electron position array xk and saves a picture in self.save_path.
        :param xk: Electron position pairs
        :return: None
        """
        xext, yext = self.xext, self.yext
        Uext = self.Uext

        fig = plt.figure(figsize=self.figsize)

        if (Uext is not None) and (xext is not None) and (yext is not None):
            Xext, Yext = np.meshgrid(xext, yext)
            plt.pcolormesh(xext * 1E6, yext * 1E6, Uext(Xext, Yext), cmap=plt.cm.RdYlBu, vmax=self.clim[1], vmin=self.clim[0])
            plt.xlim(np.min(xext) * 1E6, np.max(xext) * 1E6)
            plt.ylim(np.min(yext) * 1E6, np.max(yext) * 1E6)

        if self.coordinate_transformation is None:
            electrons_x, electrons_y = xk[::2], xk[1::2]
        else:
            r_new = self.coordinate_transformation(xk)
            electrons_x, electrons_y = r2xy(r_new)

        plt.plot(electrons_x*1E6, electrons_y*1E6, 'o', color='deepskyblue')
        plt.xlabel("$x$ ($\mu$m)")
        plt.ylabel("$y$ ($\mu$m)")
        plt.colorbar()
        plt.close(fig)

        self.monitor_convergence(xk)

    def create_movie(self, fps: int, filenames_in: str="%05d.png", filename_out: str="movie.mp4") -> None:
        """
        Generate a movie from the pictures generated by save_pictures. Movie gets saved in self.save_path
        For filenames of the type 00000.png etc use filenames_in="%05d.png".
        Files must all have the save extension and resolution.
        :param fps: frames per second (integer).
        :param filenames_in: Signature of series of file names in Unix style. Ex: "%05d.png"
        :param filename_out: File name of the output video. Ex: "movie.mp4"
        :return: None
        """
        curr_dir = os.getcwd()
        os.chdir(self.save_path)
        os.system(r"ffmpeg -r {} -b 1800 -i {} {}".format(int(fps), filenames_in, filename_out))
        os.chdir(curr_dir)

class PositionSolver:

    def __init__(self, grid_data_x: ArrayLike, grid_data_y: ArrayLike, potential_data: ArrayLike, spline_order_x: int=3, spline_order_y: int=3, 
                 smoothing: float=0, include_screening: bool=True, screening_length: float=np.inf) -> None:
        """
        This class is used for constructing the functional forms required for scipy.optimize.minimize.
        It deals with the Maxwell input data, as well as constructs the cost function used in the optimizer.
        It also calculates the gradient, that can be used to speed up the optimizer.
        :param grid_data_x: 1D array of x-data. Coordinates from grid_data_x & grid_data_y must form a rectangular grid
        :param grid_data_y: 1D array of y-data. Coordinates from grid_data_x & grid_data_y must form a rectangular grid
        :param potential_data: Energy land scape, - e V_ext.
        :param spline_order_x: Order of the interpolation in the x-direction (1 = linear, 3 = cubic)
        :param spline_order_y: Order of the interpolation in the y-direction (1 = linear, 3 = cubic)
        :param smoothing: Absolute smoothing. Effect depends on scale of potential_data.
        """
        self.interpolator = RectBivariateSpline(grid_data_x, grid_data_y, potential_data,
                                                kx=spline_order_x, ky=spline_order_y, s=smoothing)

        # Constants
        self.include_screening = include_screening
        self.screening_length = screening_length

    def V(self, xi: ArrayLike, yi: ArrayLike) -> ArrayLike:
        """
        Evaluate the electrostatic potential at coordinates xi, yi
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return: Interpolated value(s) of the data supplied to __init__ at values (xi, yi)
        """
        return self.interpolator.ev(xi, yi)

    def Velectrostatic(self, xi: ArrayLike, yi: ArrayLike) -> float:
        """
        When supplying two arrays of size n to V, it returns an array
        of size nxn, according to the meshgrid it has evaluated. We're only interested
        in the sum of the diagonal elements, so we take the sum and this represents
        the sum of the static energy of the n particles in the potential.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        return q_e * np.sum(self.V(xi, yi))

    def Vee(self, xi: ArrayLike, yi: ArrayLike, eps: float=1E-15) -> ArrayLike:
        """
        Returns the repulsion potential between two electrons separated by a distance sqrt(|xi-xj|**2 + |yi-yj|**2)
        Note the factor 1/2. in front of the potential energy to avoid overcounting.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
        np.fill_diagonal(Rij, eps)

        if self.include_screening:
            return + 1 / 2. * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-Rij/self.screening_length) / Rij
        else:
            return + 1 / 2. * q_e ** 2 / (4 * np.pi * eps0) * 1 / Rij

    def Vtotal(self, r: ArrayLike) -> float:
        """
        This can be used as a cost function for the optimizer.
        Returns the total energy of N electrons
        r is a 0D array with coordinates of the electrons.
        The x-coordinates are thus given by the even elements of r: r[::2],
        whereas the y-coordinates are the odd ones: r[1::2]
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: Scalar with the total energy of the system.
        """
        xi, yi = r[::2], r[1::2]
        Vtot = self.Velectrostatic(xi, yi)
        interaction_matrix = self.Vee(xi, yi)
        np.fill_diagonal(interaction_matrix, 0)
        Vtot += np.sum(interaction_matrix)
        return Vtot / q_e

    def dVdx(self, xi: ArrayLike, yi: ArrayLike) -> ArrayLike:
        """
        Derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=1, dy=0)

    def ddVdx(self, xi: ArrayLike, yi: ArrayLike):
        """
        Second derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=2, dy=0)

    def dVdy(self, xi: ArrayLike, yi: ArrayLike) -> ArrayLike:
        """
        Derivative of the electrostatic potential in the y-direction
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=1)

    def ddVdy(self, xi: ArrayLike, yi: ArrayLike) -> ArrayLike:
        """
        Second derivative of the electrostatic potential in the y-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=2)

    def ddVdxdy(self, xi: ArrayLike, yi: ArrayLike) -> ArrayLike:
        """
        Second derivative of the electrostatic potential in the y-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=1, dy=1)

    def grad_Vee(self, xi: ArrayLike, yi: ArrayLike, eps: float=1E-15) -> ArrayLike:
        """
        Derivative of the electron-electron interaction term
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :param eps: A small but non-zero number to avoid triggering Warning message. Exact value is irrelevant.
        :return: 1D-array of size(xi) + size(yi)
        """
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
        np.fill_diagonal(Rij, eps)

        gradx_matrix = np.zeros(np.shape(Rij))
        grady_matrix = np.zeros(np.shape(Rij))
        gradient = np.zeros(2 * len(xi))

        if self.include_screening:
            gradx_matrix = -1 * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-Rij/self.screening_length) * \
                           (Xi - Xj) * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
            grady_matrix = +1 * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-Rij/self.screening_length) * \
                           (Yi - Yj) * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
        else:
            gradx_matrix = -1 * q_e ** 2 / (4 * np.pi * eps0) * (Xi - Xj) / Rij ** 3
            grady_matrix = +1 * q_e ** 2 / (4 * np.pi * eps0) * (Yi - Yj) / Rij ** 3


        np.fill_diagonal(gradx_matrix, 0)
        np.fill_diagonal(grady_matrix, 0)

        gradient[::2] = np.sum(gradx_matrix, axis=0)
        gradient[1::2] = np.sum(grady_matrix, axis=0)

        return gradient

    def grad_total(self, r: ArrayLike) -> float:
        """
        Total derivative of the cost function. This may be used in the optimizer to converge faster.
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: 1D array of length len(r), where grad_total = np.array([dV/dx|r0, dV/dy|r0, ...])
        """
        xi, yi = r[::2], r[1::2]
        gradient = np.zeros(len(r))
        gradient[::2] = self.dVdx(xi, yi)
        gradient[1::2] = self.dVdy(xi, yi)
        gradient += self.grad_Vee(xi, yi) / q_e
        return gradient

    def thermal_kick_x(self, x: ArrayLike, y: ArrayLike, T: float, maximum_dx: Optional[float]=None) -> float:
        ktrapx = np.abs(q_e * self.ddVdx(x, y))
        ret = np.sqrt(2 * kB * T / ktrapx)
        if maximum_dx is not None:
            ret[ret > maximum_dx] = maximum_dx
            return ret
        else:
            return ret

    def thermal_kick_y(self, x: ArrayLike, y: ArrayLike, T: float, maximum_dy: Optional[float]=None) -> float:
        ktrapy = np.abs(q_e * self.ddVdy(x, y))
        ret = np.sqrt(2 * kB * T / ktrapy)
        if maximum_dy is not None:
            ret[ret > maximum_dy] = maximum_dy
            return ret
        else:
            return ret

    def single_thread(self, iteration, electron_initial_positions, T, cost_function, minimizer_dict, maximum_dx, maximum_dy):
        xi, yi = r2xy(electron_initial_positions)
        np.random.seed(np.int(time.time()) + iteration)
        xi_prime = xi + self.thermal_kick_x(xi, yi, T, maximum_dx=maximum_dx) * np.random.randn(len(xi))
        yi_prime = yi + self.thermal_kick_y(xi, yi, T, maximum_dy=maximum_dy) * np.random.randn(len(yi))
        electron_perturbed_positions = xy2r(xi_prime, yi_prime)
        return minimize(cost_function, electron_perturbed_positions, **minimizer_dict)

    def parallel_perturb_and_solve(self, cost_function: callable, N_perturbations: int, T: float, 
                                   solution_data_reference: dict, minimizer_dict: dict,
                                   maximum_dx: Optional[float]=None, maximum_dy: Optional[float]=None) -> dict:
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        This function runs N_perturbations in parallel on the cores of your CPU.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_dict: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """
        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference
        pool = multiprocessing.Pool()

        tasks = []
        iteration = 0
        while iteration < N_perturbations:
            iteration += 1
            tasks.append((iteration, electron_initial_positions, T, cost_function, minimizer_dict, maximum_dx, maximum_dy,))

        results = [pool.apply_async(self.single_thread, t) for t in tasks]
        for result in results:
            res = result.get()

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                #cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res

        # Nothing has changed by perturbing the reference solution
        if (best_result['x'] == solution_data_reference['x']).all():
            cprint("Solution data unchanged after perturbing", "white")
        # Or there is a new minimum
        else:
            cprint("Better solution found (%.3f%% difference)" \
                   % (100 * (best_result['fun'] - solution_data_reference['fun']) / solution_data_reference['fun']),
                   "green")


        return best_result

    def perturb_and_solve(self, cost_function: callable, N_perturbations: int, T: float, solution_data_reference: dict,
                          maximum_dx: Optional[float]=None, maximum_dy: Optional[float]=None, do_print: bool=True, 
                          **minimizer_options) -> dict:
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_options: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """
        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference

        for n in range(N_perturbations):
            xi, yi = r2xy(electron_initial_positions)
            xi_prime = xi + self.thermal_kick_x(xi, yi, T, maximum_dx=maximum_dx) * np.random.randn(len(xi))
            yi_prime = yi + self.thermal_kick_y(xi, yi, T, maximum_dy=maximum_dy) * np.random.randn(len(yi))
            electron_perturbed_positions = xy2r(xi_prime, yi_prime)

            res = minimize(cost_function, electron_perturbed_positions, **minimizer_options)

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                if do_print:
                    cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res
            elif res['status'] == 0 and res['fun'] > best_result['fun']:
                pass  # No new minimum was found after perturbation, this is quite common.
            elif res['status'] != 0 and res['fun'] < best_result['fun']:
                if do_print:
                    cprint("\tThere is a lower state, but minimizer didn't converge!", "red")
            elif res['status'] != 0 and res['fun'] > best_result['fun']:
                pass

        return best_result

    def calculate_mu(self, ri: ArrayLike) -> ArrayLike:
        electrons_x, electrons_y = r2xy(ri)
        interactions = self.Vee(electrons_x, electrons_y) / q_e
        np.fill_diagonal(interactions, 0)
        mu = list()
        el = 0
        for electron_x, electron_y in zip(electrons_x, electrons_y):
            mu.append(np.sum(interactions[el, :]) + self.Velectrostatic(electron_x, electron_y) / 1.602E-19)
            el += 1

        return np.array(mu)