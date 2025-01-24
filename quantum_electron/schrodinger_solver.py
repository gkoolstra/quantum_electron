import scipy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from .utils import construct_symmetric_y, make_potential, find_minimum_location, PotentialVisualization
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy.linalg import eig
from scipy.constants import hbar, m_e, elementary_charge as q_e
from typing import Dict, List, Optional
from numpy.typing import ArrayLike
from itertools import product


class Schrodinger:
    """Abstract class for solving the 1D and 2D Schrodinger equation 
    using finite differences and sparse matrices"""

    def __init__(self, sparse_args=None, solve=True):
        """ @param sparse_args arguments for the eigsh sparse solver
            @param solve if solve=True then it will immediately construct the Hamiltonian and solve for the eigenvalues
        """
        self.solved = False
        self.sparse_args = sparse_args
        self.solved = False
        if solve:
            self.solve()

    @staticmethod
    def uv(vec):
        """normalizes a vector
            @param vec vector to normalize
        """
        return vec / np.sqrt(np.dot(vec, vec))

    @staticmethod
    def Dmat(numpts, delta=1):
        """Derivative matrix
            @param numpts dimension of derivative matrix
            @param delta optional scaling of point spacing
        """
        a = 0.5 / delta * np.ones(numpts)
        a[0] = 0
        a[-2] = 0
        # b=-2./delta**2*ones(numpts); b[0]=0;b[-1]=0
        c = -0.5 / delta * np.ones(numpts)
        c[1] = 0
        c[-1] = 0
        return sparse.spdiags([a, c], [-1, 1], numpts, numpts)

    @staticmethod
    def D2mat(numpts, delta=1, periodic=True, q=0):
        """2nd Derivative matrix
            @param numpts dimension of derivative matrix
            @param delta spacing between points
            @param periodic whether derivative wraps around (default True) 
            @param q is a quasimomentum between -pi and pi, which is used if periodic=True
        """

        a = 1. / delta ** 2 * np.ones(numpts)
        b = -2. / delta ** 2 * np.ones(numpts)
        c = 1. / delta ** 2 * np.ones(numpts)
        # print "delta = %f" % (delta)
        if periodic:
            if q == 0:
                return sparse.spdiags([c, a, b, c, c], [-numpts + 1, -1, 0, 1, numpts - 1], numpts, numpts)
            else:
                return sparse.spdiags([np.exp(-(0. + 1.j) * q) * c, a, b, c, np.exp((0. + 1.j) * q) * c],
                                      [-numpts + 1, -1, 0, 1, numpts - 1], numpts, numpts)
        else:
            return sparse.spdiags([a, b, c], [-1, 0, 1], numpts, numpts)

    def Hamiltonian(self):
        """Abstract method used by solver"""
        return None

    def solve(self, sparse_args=None):
        """Constructs and solves for eigenvalues and eigenvectors of Hamiltonian
            @param sparse_args if present used in eigsh sparse solver"""
        Hmat = self.Hamiltonian()
        if sparse_args is not None:
            self.sparse_args = sparse_args
        if self.sparse_args is None:
            en, ev = eig(Hmat.todense())
        else:
            en, ev = eigsh(Hmat, **self.sparse_args)
        ev = np.transpose(np.array(ev))[np.argsort(en)]
        en = np.sort(en)
        self.en = en
        self.ev = ev
        self.solved = True
        return self.en, self.ev

    def energies(self, num_levels=-1):
        """returns eigenvalues of Hamiltonian (solves if not already solved)"""
        if not self.solved:
            self.solve()
        return self.en[:num_levels]

    def psis(self, num_levels=-1):
        """returns eigenvectors of Hamiltonian (solves if not already solved)"""
        if not self.solved:
            self.solve()
        return self.ev[:num_levels]

    def reduced_operator(self, operator, num_levels=-1):
        """Finds operator in eigenbasis of the hamiltonian truncated to num_levels
        @param operator a (sparse) matrix representing an operator in the x basis
        @num_levels number of levels to truncate Hilbert space
        """
        if not self.solved:
            self.solve()
        if sparse.issparse(operator):
            return np.array([np.array([np.dot(psi1, operator.dot(psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
                             self.psis(num_levels)])
        else:
            return np.array([np.array([np.dot(psi1, np.dot(operator, psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
                             self.psis(num_levels)])


class Schrodinger2D(Schrodinger):
    def __init__(self, x, y, U, KEx=1, KEy=1, periodic_x=False, periodic_y=False, qx=0, qy=0, sparse_args=None,
                 solve=True):
        """@param x is array of locations in x direction
           @param y is array of locations in y direction
           @param U is array of potential at x
           @param KEx is kinetic energy prefactor in x direction
           @param KEy is kinetic energy prefactor in y direction
           @param periodic_x True/False for x boundary conditions
           @param periodic_y True/False for y boundary conditions
           @param qx, if periodic_x=True then use exp(i qx) for boundary condition phase
           @param qy, if periodic_y=True then use exp(i qy) for boundary condition phase
           @param num_levels (None)...number of levels for sparse solver or None for dense solve...sparse not working right yet...+
        """
        self.x = x
        self.y = y
        self.U = U
        self.KEx = KEx
        self.KEy = KEy
        self.qx = qx
        self.qy = qy
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        Schrodinger.__init__(self, sparse_args=sparse_args, solve=solve)

    def Hamiltonian(self):
        """Constructs Hamiltonian using the potential and Kinetic energy terms"""
        U = self.U.flatten()
        Vmat = sparse.spdiags([U], [0], len(U), len(U))
        Kmat = sparse.kron(-self.KEy * Schrodinger.D2mat(len(self.y), self.y[1] - self.y[0], self.periodic_y, self.qy),
                           sparse.identity(len(self.x))) + \
            sparse.kron(sparse.identity(len(self.y)),
                        -self.KEx * Schrodinger.D2mat(len(self.x), self.x[1] - self.x[0], self.periodic_x, self.qx))
        return Kmat + Vmat

    def get_2Dpsis(self, num_levels=-1):
        psis = []
        for psi in self.psis(num_levels):
            psis.append(np.reshape(psi, (len(self.y), len(self.x))))
        return psis

    def plot(self, num_levels=10):
        """Plots potential, energies, and wavefunctions
        @param num_levels (-1 by default) number of levels to plot"""
        if num_levels == -1:
            num_levels = len(self.energies())
        print(self.energies(num_levels))
        plt.figure(figsize=(20, 5))
        plt.subplot(1, num_levels + 1, 1)
        self.plot_potential()
        # xlabel('$\phi$')
        for ii, psi2D in enumerate(self.get_2Dpsis(num_levels)):
            plt.subplot(1, num_levels + 1, ii + 2)
            # imshow(psi2D.real,extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]),interpolation="None",aspect='auto')
            plt.imshow(psi2D.real, interpolation="None", aspect='auto')
            plt.xlabel(ii)

    def plot_potential(self):
        """Plots potential energy landscape"""
        plt.imshow(self.U, extent=(
            self.x[0], self.x[-1], self.y[0], self.y[-1]), aspect='auto', interpolation='None')
        plt.xlabel('x')
        plt.ylabel('y')


class SingleElectron(Schrodinger2D):
    def __init__(self, x, y, potential_function, sparse_args=None, solve=True):
        """
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/generated/scipy.sparse.linalg.eigsh.html
        potential_function: a function that takes as arguments a meshgrid of x, y coordinates. For a positive voltage on the
        electrodes, this function is negative!
        """
        self.x = x
        self.y = y

        self.numxpts = len(x)
        self.numypts = len(y)

        self.potential = potential_function

        Vxy = self.evaluate_potential(self.x, self.y)
        Schrodinger2D.__init__(self, x=self.x, y=self.y, U=Vxy, KEx=1, KEy=1,
                               periodic_x=False, periodic_y=False, qx=0, qy=0,
                               sparse_args=sparse_args, solve=solve)

    def evaluate_potential(self, x, y):
        X, Y = np.meshgrid(x, y)
        return + 2 * m_e * q_e * self.potential((X, Y)) / hbar ** 2

    def sparsify(self, num_levels=10):
        self.U = self.evaluate_potential(self.x, self.y)
        self.sparse_args = {'k': num_levels,  # Find k eigenvalues and eigenvectors
                            # ‘LM’ : Largest (in magnitude) eigenvalues
                            'which': 'LM',
                            # 'sigma' : Find eigenvalues near sigma using shift-invert mode.
                            'sigma': np.min(self.U),
                            'maxiter': None}  # Maximum number of Arnoldi update iterations allowed Default: n*10


class QuantumAnalysis(PotentialVisualization):
    """This class solves the Schrodinger equation for a single electron on helium. Typical workflow: 

    qa = QuantumAnalysis(potential_dict=potential_dict, voltage_dict=voltage_dict)
    qa.get_quantum_spectrum(coor=None, dxdy=[.8, .8])
    """

    def __init__(self, potential_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, float]):
        """Class for solving quantum properties of a single electron trapped in a dot

        Args:
            potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
        """
        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict
        self.solved = False

        PotentialVisualization.__init__(
            self, potential_dict=potential_dict, voltages=voltage_dict)

    def update_voltages(self, voltage_dict: Dict[str, float]):
        """Update the voltage dictionary

        Args:
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
        """
        self.voltage_dict = voltage_dict
        self.solved = False

    def solve_system(self, coor: List[float] = [0, 0], dxdy: List[float] = [1, 2], N_evals: float = 10, n_x: int = 150, n_y: int = 100) -> None:
        """Solve the Schrodinger equation for a given set of voltages.

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            N_evals (float, optional): Number of eigenvalues to consider. Defaults to 10.
        """
        # If not specified as a function argument, coor will be the minimum of the potential
        if coor is None:
            coor = find_minimum_location(
                self.potential_dict, self.voltage_dict)

        # Note that xsol and ysol determine the x and y points for which you want to solve the Schrodinger equation
        self.xsol = np.linspace(
            coor[0]-dxdy[0]/2, coor[0]+dxdy[0]/2, n_x) * 1e-6
        y_symmetric = construct_symmetric_y(coor[1]-dxdy[1]/2, n_y) * 1e-6
        self.ysol = np.zeros(2 * len(y_symmetric))
        self.ysol[:len(y_symmetric)] = y_symmetric
        self.ysol[len(y_symmetric):] = -y_symmetric[::-1]

        potential = make_potential(self.potential_dict, self.voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        potential_function = scipy.interpolate.RegularGridInterpolator((self.potential_dict['xlist']*1e-6,
                                                                        self.potential_dict['ylist']*1e-6),
                                                                       -potential)

        # Note that the solution is sampled over the arrays xsol, ysol which can be set indepently from the FEM x and y points.
        se = SingleElectron(self.xsol, self.ysol,
                            potential_function=potential_function, solve=False)
        se.sparsify(num_levels=N_evals)
        Evals, Evecs = se.solve(sparse_args=se.sparse_args)

        self.Psis = se.get_2Dpsis(N_evals)
        self.mode_frequencies = (
            Evals - Evals[0]) * hbar**2 / (2 * q_e * m_e) * q_e / (2 * np.pi * hbar)

        self.solved = True

    def classify_wavefunction_by_well(self) -> ArrayLike:
        """This function classifies the wavefunctions by well. If the potential has a double well, the wave function will be marked with +1 or -1. 
        If there is a well it's assumed to be in the y-direction, and +1 is associated with positive y and -1 with negative. 0 is a single well.

        Returns:
            ArrayLike: array with the same length as Psis.
        """
        assert self.solved is True, print(
            "You must solve the Schrodinger equation first!")

        # classify by finding the center of mass of the wave function
        X, Y = np.meshgrid(self.xsol, self.ysol)

        well_classification = list()
        for k in range(len(self.Psis)):
            y_com = np.mean(np.abs(self.Psis[k])
                            * Y) / np.mean(np.abs(self.Psis[k]))
            x_com = np.mean(np.abs(self.Psis[k])
                            * X) / np.mean(np.abs(self.Psis[k]))

            if y_com > 0.1e-6:
                well_classification.append(+1)
            elif y_com < -0.1e-6:
                well_classification.append(-1)
            else:
                well_classification.append(0)

        return np.array(well_classification)

    def classify_wavefunction_by_xy(self) -> List:
        """Classifies the wave function by labeling it with a number nx and ny. These numbers capture the number of crests of the wave function in 
        the x and y direction, respectively. 

        Returns:
            List: List of dictionaries. The length of this list is equal to the length of Psis.
        """
        assert self.solved is True, print(
            "You must solve the Schrodinger equation first!")

        classification = list()
        for k in range(len(self.Psis)):

            sig = np.sum(self.Psis[k] ** 2, axis=0)
            n_x = len(scipy.signal.find_peaks(sig, height=np.max(sig)/2)[0])

            sig = np.sum(self.Psis[k] ** 2, axis=1)
            n_y = len(scipy.signal.find_peaks(sig, height=np.max(sig)/2)[0])

            classification.append({"nx": n_x - 1,
                                   "ny": n_y - 1})

        return classification

    def classification_to_latex(self, classification: dict) -> str:
        """This function takes the classification dictionary and transforms it into a string for plotting.

        Args:
            classification (dict): Dictionary with elements 'nx' and 'ny' (both integers)

        Returns:
            _type_: String for use in matplotlib legends, titles, etc.
        """
        return fr"$|{classification['nx']:d}_x {classification['ny']:d}_y \rangle$"

    def get_quantum_spectrum(self, coor: Optional[List[float]] = [0, 0], dxdy: List[float] = [1, 2], plot_wavefunctions: bool = False,
                             axes_zoom: Optional[float] = None, **solve_kwargs) -> tuple[ArrayLike, ArrayLike]:
        """Returns the frequencies of the first N eigenmodes for a single electron trapped in a potential. 

        Args:
            potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            voltages (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            plot_wavefunctions (bool, optional): Whether to plot the wave functions or simply return the frequencies. Defaults to False.
            axes_zoom (Optional[float], optional): Axes extent around the wavefunction. If None the axes are set by coor and dxdy. Defaults to None.

        Returns:
            tuple[ArrayLike, ArrayLike]: Eigenfrequencies of the first N motional modes in Hz, and a classification of the mode.
        """
        if coor is None:
            coor = find_minimum_location(self.potential_dict, self.voltage_dict)

        if not self.solved:
            self.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)

        if plot_wavefunctions:
            fig = plt.figure(figsize=(12., 6.))

        well_classification = self.classify_wavefunction_by_well()
        xy_classification = self.classify_wavefunction_by_xy()

        for k in range(6):
            if plot_wavefunctions:
                plt.subplot(2, 3, k+1)
                plt.pcolormesh(self.xsol/1e-6, self.ysol/1e-6, self.Psis[k], cmap=plt.cm.RdBu_r,
                               vmin=-np.max(np.abs(self.Psis[k])),
                               vmax=np.max(np.abs(self.Psis[k])))
                cbar = plt.colorbar()
                tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
                cbar.locator = tick_locator
                cbar.update_ticks()

            if plot_wavefunctions:
                zdata = -make_potential(self.potential_dict,
                                        self.voltage_dict).T
                contours = [np.round(np.min(zdata), 3) +
                            k*1e-3 for k in range(5)]
                CS = plt.contour(
                    self.potential_dict['xlist'], self.potential_dict['ylist'], zdata, levels=contours)
                plt.gca().clabel(CS, CS.levels, inline=True, fontsize=10)
                plt.title(rf"{self.classification_to_latex(xy_classification[k])} " +
                          f"({well_classification[k]} well) - {self.mode_frequencies[k]/1e9:.2f} GHz", size=10)

                if axes_zoom is not None:
                    # classify by finding the center of mass of the wave function
                    X, Y = np.meshgrid(self.xsol, self.ysol)
                    y_com = np.mean(
                        np.abs(self.Psis[k]) * Y) / np.mean(np.abs(self.Psis[k]))
                    x_com = np.mean(
                        np.abs(self.Psis[k]) * X) / np.mean(np.abs(self.Psis[k]))

                    plt.xlim((x_com/1e-6 - axes_zoom/2),
                             (x_com/1e-6 + axes_zoom/2))
                    plt.ylim((y_com/1e-6 - axes_zoom/2),
                             (y_com/1e-6 + axes_zoom/2))
                else:
                    plt.xlim(np.min(self.xsol/1e-6), np.max(self.xsol/1e-6))
                    plt.ylim(np.min(self.ysol/1e-6), np.max(self.ysol/1e-6))

                plt.locator_params(axis='both', nbins=4)

                if k >= 3:
                    plt.xlabel("$x$"+f" ({chr(956)}m)")

                if not k % 3:
                    plt.ylabel("$y$"+f" ({chr(956)}m)")

        if plot_wavefunctions:
            fig.tight_layout()

        return self.mode_frequencies

    def get_anharmonicity(self) -> float:
        """Calculate the anharmonicity. The anharmonicity here is defined as (f|0x2y> - f|0x1y>) - (f|0x1y> - f|0x0y>) 

        Returns:
            float: Anharmonicity in Hz.
        """
        assert self.solved is True, print(
            "You must solve the Schrodinger equation first!")

        frequencies = self.mode_frequencies
        classifications = self.classify_wavefunction_by_xy()

        f_2y = frequencies[classifications.index({'nx': 0, 'ny': 2})]
        f_1y = frequencies[classifications.index({'nx': 0, 'ny': 1})]
        try:
            f_0y = frequencies[classifications.index({'nx': 0, 'ny': 0})]
        except:
            # In some pathological cases the ground state is spread out over two wells, and it's not recognized. Then we can assume it's the first index.
            f_0y = frequencies[0]

        anharmonicity = (f_2y - f_1y) - (f_1y - f_0y)

        return anharmonicity

    def get_resonator_coupling(self, coor: Optional[List[float]] = [0, 0], dxdy: List[float] = [1, 2], Ex: float = 0, Ey: float = 1e6, resonator_impedance: float = 50,
                               resonator_frequency: float = 4e9, plot_result: bool = True, **solve_kwargs) -> ArrayLike:
        """Calculate the coupling strength in Hz for mode |i> to mode |j>

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            Ex (float, optional): Electric field of the relevant microwave mode in the x-direction. Defaults to 0.
            Ey (float, optional): Electric field of the relevant microwave mode in the y-direction. Defaults to 1e6.
            resonator_impedance (float, optional): Resonator impedance in ohms. Defaults to 50.
            resonator_frequency (float, optional): Resonator frequency in Hz. Defaults to 4e9.
            plot_result (bool, optional): Plots the matrix. Defaults to True.

        Returns:
            ArrayLike: The g_ij matrix
        """

        if not self.solved:
            self.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)

        N_evals = len(self.Psis)

        # The resonator coupling is a symmetric matrix
        g_ij = np.zeros((N_evals, N_evals))
        X, Y = np.meshgrid(self.xsol, self.ysol)

        prefactor = q_e * np.sqrt(hbar * (2 * np.pi * resonator_frequency)
                                  ** 2 * resonator_impedance / 2) * 1 / (2 * np.pi * hbar)

        for i in range(N_evals):
            for j in range(N_evals):
                g_ij[i, j] = prefactor * \
                    np.sum(self.Psis[i] * (X * Ex + Y * Ey)
                           * np.conjugate(self.Psis[j]))

        if plot_result:
            fig = plt.figure(figsize=(7., 4.))
            plt.imshow(np.abs(g_ij)/1e6, cmap=plt.cm.Blues)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r"Coupling strength $g_{ij} / 2\pi$ (MHz)")
            plt.xlabel("Mode index $j$")
            plt.ylabel("Mode index $i$")

            for (i, j) in product(range(N_evals), range(N_evals)):
                g_value = np.abs(g_ij[i, j] / 1e6)
                if g_value > 0.2:
                    col = 'white' if g_value > np.max(
                        np.abs(g_ij)) / 1e6 / 2 else 'black'
                    plt.text(i, j, f"{g_ij[i, j]/ 1e6:.1f}",
                             size=9, ha='center', va='center', color=col)

        return g_ij
