import scipy
import numpy as np
from .utils import r2xy
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e

from numpy.typing import ArrayLike
from typing import List, Dict

class EOMSolver:
    def __init__(self, f0: float, Z0: float, Ex: callable, Ey: callable, curv_xx: callable, curv_xy: callable, curv_yy: callable) -> None:
        self.f0 = f0
        self.Z0 = Z0
        
        self.Ex = Ex
        self.Ey = Ey
        self.curv_xx = curv_xx 
        self.curv_xy = curv_xy
        self.curv_yy = curv_yy
    
    def setup_eom(self, ri: ArrayLike) -> tuple[ArrayLike]:
        """
        Set up the Matrix used for determining the electron frequency.
        :param electron_positions: Electron positions, in the form [x0, y0, x1, y1, ...]
        :return: M^(-1) * K
        """
        omega0 = 2 * np.pi * self.f0
        L = self.Z0 / omega0
        C = 1 / (omega0**2 * L)

        num_electrons = int(len(ri) / 2)
        xe, ye = r2xy(ri)

        # Set up the inverse of the mass matrix first
        diag_invM = 1 / m_e * np.ones(2 * num_electrons + 1)
        diag_invM[0] = 1 / L
        invM = np.diag(diag_invM)
        M = np.diag(np.array([L] + [m_e] * (2 * num_electrons)))

        # Set up the kinetic matrix next
        Kij_plus, Kij_minus, Lij = np.zeros(np.shape(invM)), np.zeros(np.shape(invM)), np.zeros(np.shape(invM))
        K = np.zeros((2 * num_electrons + 1, 2 * num_electrons + 1))
        # Row 1 and column 1 only have bare cavity information, and cavity-electron terms
        K[0, 0] = 1 / C
        K[1:num_electrons+1, 0] = K[0, 1:num_electrons+1] = q_e / C * self.Ex(xe, ye)
        K[num_electrons+1:2*num_electrons+1, 0] = K[0, num_electrons+1:2*num_electrons+1] = q_e / C * self.Ey(xe, ye)

        kij_plus = np.zeros((num_electrons, num_electrons))
        kij_minus = np.zeros((num_electrons, num_electrons))
        lij = np.zeros((num_electrons, num_electrons))

        Xi, Yi = np.meshgrid(xe, ye)
        Xj, Yj = Xi.T, Yi.T
        XiXj = Xi - Xj
        YiYj = Yi - Yj
        rij = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
        np.fill_diagonal(XiXj, 1E-15)
        tij = np.arctan(YiYj / XiXj)

        # Remember to set the diagonal back to 0
        np.fill_diagonal(tij, 0)
        # We'll be dividing by rij, so to avoid raising warnings:
        np.fill_diagonal(rij, 1E-15)

        if self.screening_length == np.inf:
            # print("Coulomb!")
            # Note that an infinite screening length corresponds to the Coulomb case. Usually it should be twice the
            # helium depth
            kij_plus = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * (1 + 3 * np.cos(2 * tij)) / rij ** 3
            kij_minus = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * (1 - 3 * np.cos(2 * tij)) / rij ** 3
            lij = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * 3 * np.sin(2 * tij) / rij ** 3
        else:
            # print("Yukawa!")
            rij_scaled = rij / self.screening_length
            kij_plus = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-rij_scaled) / rij ** 3 * \
                              (1 + rij_scaled + rij_scaled ** 2 + (3 + 3 * rij_scaled + rij_scaled ** 2) * np.cos(
                                  2 * tij))
            kij_minus = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-rij_scaled) / rij ** 3 * \
                              (1 + rij_scaled + rij_scaled ** 2 - (3 + 3 * rij_scaled + rij_scaled ** 2) * np.cos(
                                  2 * tij))
            lij = 1 / 4. * q_e ** 2 / (4 * np.pi * eps0) * np.exp(-rij_scaled) / rij ** 3 * \
                              (3 + 3 * rij_scaled + rij_scaled ** 2) * np.sin(2 * tij)

        np.fill_diagonal(kij_plus, 0)
        np.fill_diagonal(kij_minus, 0)
        np.fill_diagonal(lij, 0)

        # Note: not sure where the factor 2 comes from
        Kij_plus = -kij_plus + np.diag(q_e*self.curv_xx(xe, ye) + np.sum(kij_plus, axis=1))
        Kij_minus = -kij_minus + np.diag(q_e*self.curv_yy(xe, ye) + np.sum(kij_minus, axis=1))
        Lij = -lij + np.diag(q_e*self.curv_xy(xe, ye) + np.sum(lij, axis=1))

        K[1:num_electrons+1,1:num_electrons+1] = Kij_plus
        K[num_electrons+1:2*num_electrons+1, num_electrons+1:2*num_electrons+1] = Kij_minus
        K[1:num_electrons+1, num_electrons+1:2*num_electrons+1] = Lij
        K[num_electrons+1:2*num_electrons+1, 1:num_electrons+1] = Lij

        return K, M

    def solve_eom(self, LHS: ArrayLike, RHS: ArrayLike) -> tuple[ArrayLike]:
        """
        Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
        :param LHS: matrix product of M^(-1) K
        :return: Eigenvalues, Eigenvectors
        """
        # EVals, EVecs = np.linalg.eig(np.dot(np.linalg.inv(RHS), LHS))
        EVals, EVecs = scipy.linalg.eigh(LHS, b=RHS)
        return EVals, EVecs