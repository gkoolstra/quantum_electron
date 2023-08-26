import scipy
import numpy as np
from matplotlib import pyplot as plt
from .utils import construct_symmetric_y
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import pi, linspace, cos, sin, ones, transpose, reshape, array, argsort, sort, \
    meshgrid, amax, amin, dot, sqrt, exp, tanh, sign, argmax
from numpy.linalg import eig
from scipy.constants import hbar, m_e, elementary_charge as q_e

m_e = 9.11e-31
q_e = 1.602e-19
hbar = 1.055e-34

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
        if solve: self.solve()

    @staticmethod
    def uv(vec):
        """normalizes a vector
            @param vec vector to normalize
        """
        return vec / sqrt(dot(vec, vec))

    @staticmethod
    def Dmat(numpts, delta=1):
        """Derivative matrix
            @param numpts dimension of derivative matrix
            @param delta optional scaling of point spacing
        """
        a = 0.5 / delta * ones(numpts)
        a[0] = 0
        a[-2] = 0
        #b=-2./delta**2*ones(numpts); b[0]=0;b[-1]=0
        c = -0.5 / delta * ones(numpts)
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

        a = 1. / delta ** 2 * ones(numpts)
        b = -2. / delta ** 2 * ones(numpts)
        c = 1. / delta ** 2 * ones(numpts)
        #print "delta = %f" % (delta)
        if periodic:
            if q == 0:
                return sparse.spdiags([c, a, b, c, c], [-numpts + 1, -1, 0, 1, numpts - 1], numpts, numpts)
            else:
                return sparse.spdiags([exp(-(0. + 1.j) * q) * c, a, b, c, exp((0. + 1.j) * q) * c],
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
        if sparse_args is not None: self.sparse_args = sparse_args
        if self.sparse_args is None:
            en, ev = eig(Hmat.todense())
        else:
            en, ev = eigsh(Hmat, **self.sparse_args)
        ev = transpose(array(ev))[argsort(en)]
        en = sort(en)
        self.en = en
        self.ev = ev
        self.solved = True
        return self.en, self.ev

    def energies(self, num_levels=-1):
        """returns eigenvalues of Hamiltonian (solves if not already solved)"""
        if not self.solved: self.solve()
        return self.en[:num_levels]

    def psis(self, num_levels=-1):
        """returns eigenvectors of Hamiltonian (solves if not already solved)"""
        if not self.solved: self.solve()
        return self.ev[:num_levels]

    def reduced_operator(self, operator, num_levels=-1):
        """Finds operator in eigenbasis of the hamiltonian truncated to num_levels
        @param operator a (sparse) matrix representing an operator in the x basis
        @num_levels number of levels to truncate Hilbert space
        """
        if not self.solved: self.solve()
        if sparse.issparse(operator):
            return array([array([dot(psi1, operator.dot(psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
                          self.psis(num_levels)])
        else:
            return array([array([dot(psi1, dot(operator, psi2)) for psi2 in self.psis(num_levels)]) for psi1 in
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
            psis.append(reshape(psi, (len(self.y), len(self.x))))
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
        #xlabel('$\phi$')
        for ii, psi2D in enumerate(self.get_2Dpsis(num_levels)):
            plt.subplot(1, num_levels + 1, ii + 2)
            #imshow(psi2D.real,extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]),interpolation="None",aspect='auto')
            plt.imshow(psi2D.real, interpolation="None", aspect='auto')
            plt.xlabel(ii)

    def plot_potential(self):
        """Plots potential energy landscape"""
        plt.imshow(self.U, extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]), aspect='auto', interpolation='None')
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
                            'which': 'LM',  # ‘LM’ : Largest (in magnitude) eigenvalues
                            'sigma': np.min(self.U),  # 'sigma' : Find eigenvalues near sigma using shift-invert mode.
                            'maxiter': None}  # Maximum number of Arnoldi update iterations allowed Default: n*10
        
def get_quantum_spectrum(potential_dict, voltages, plot_wavefunctions=False):
    # potential_dict --> xlist, ylist, electrode 1, electrode 2, etc
    # voltages --> electrode 1, electrode 2, etc.
    # xsol and ysol determine the x and y points for which you want to solve the Schrodinger equation
    xsol = np.linspace(-1.0e-6, -0.0e-6, 151)
    y_symmetric = construct_symmetric_y(-0.5e-6, 50)
    ysol = np.zeros(2 * len(y_symmetric))
    ysol[:len(y_symmetric)] = y_symmetric
    ysol[len(y_symmetric):] = -y_symmetric[::-1]

    # potential_dict has the same keys as voltages, plus extra keys xlist and ylist, which contain the xpoints and ypoints (the potential points in units of um)
    for k, key in enumerate(list(voltages.keys())):
        if k == 0: 
            potential = potential_dict[key] * voltages[key] 
        else:
            potential += potential_dict[key] * voltages[key]

    potential_function = scipy.interpolate.RegularGridInterpolator((potential_dict['xlist']*1e-6, potential_dict['ylist']*1e-6), -potential)

    N_evals = 10

    se = SingleElectron(xsol, ysol, potential_function=potential_function, solve=False)
    se.sparsify(num_levels=N_evals)
    Evals, Evecs = se.solve(sparse_args=se.sparse_args)

    Psis = se.get_2Dpsis(N_evals)
    mode_frequencies = (Evals - Evals[0]) * hbar**2 / (2 * q_e * m_e) * q_e / (2 * np.pi * hbar)

    zdata = -potential.T

    classification = list()
    if plot_wavefunctions:
        plt.figure(figsize=(12.,6.))

    for k in range(6):
        if plot_wavefunctions:
            plt.subplot(2, 3, k+1)
            plt.pcolormesh(xsol/1e-6, ysol/1e-6, Psis[k], cmap=plt.cm.RdBu_r, vmin=-.2, vmax=0.2)
            plt.colorbar()
        
        # classify by finding the center of mass of the wave function
        X, Y = np.meshgrid(xsol, ysol)
        y_com = np.mean(np.abs(Psis[k]) * Y) / np.mean(np.abs(Psis[k]))
        
        if y_com > 0.1e-6:
            well = +1
        elif y_com < -0.1e-6:
            well = -1
        else: 
            well = 0
        
        if plot_wavefunctions:
            contours = [np.round(np.min(zdata), 3) +k*1e-3 for k in range(5)]
            CS = plt.contour(potential_dict['xlist'], potential_dict['ylist'], zdata, levels=contours)
            plt.gca().clabel(CS, CS.levels, inline=True, fontsize=10)
            plt.text(-0.90, 0.4, f"({well} well) - {mode_frequencies[k]/1e9:.2f} GHz")
            
            plt.xlim(np.min(xsol)/1e-6, np.max(xsol)/1e-6)
            plt.ylim(np.min(ysol)/1e-6, np.max(ysol)/1e-6)
        
            if k >= 3:
                plt.xlabel("$x$"+r" ($\mu$m)")
                
            if not k%3:
                plt.ylabel("$y$"+r" ($\mu$m)")
            
        classification.append(well)
                
    return mode_frequencies, np.array(classification)

def plot_potential(potential_dict, voltages):
    # potential_dict --> xlist, ylist, electrode 1, electrode 2, etc
    # voltages --> electrode 1, electrode 2, etc.
    for k, key in enumerate(list(voltages.keys())):
        if k == 0: 
            potential = potential_dict[key] * voltages[key] 
        else:
            potential += potential_dict[key] * voltages[key]

    zdata = -potential.T

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    plt.pcolormesh(potential_dict['xlist'], potential_dict['ylist'], zdata, cmap=plt.cm.RdYlBu_r)
    plt.colorbar()
    
    xidx, yidx = np.unravel_index(zdata.argmin(), zdata.shape)
    plt.plot(potential_dict['xlist'][yidx], potential_dict['ylist'][xidx], '*', color='white')

    zoom = 1

    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)

    ax.set_aspect('equal')

    contours = [np.round(np.min(zdata), 3) +k*1e-3 for k in range(5)]
    CS = plt.contour(potential_dict['xlist'], potential_dict['ylist'], zdata, levels=contours)
    ax.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.xlabel("x")
    plt.ylabel("y")