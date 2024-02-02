import scipy
import numpy as np
from .utils import r2xy
from scipy.constants import elementary_charge as q_e, epsilon_0 as eps0, electron_mass as m_e

from numpy.typing import ArrayLike
from typing import List, Dict, Optional
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import patheffects as pe
from IPython import display 

class EOMSolver:
    def __init__(self, f0: float, Z0: float, Ex: callable, Ey: callable, Ex_up: callable, Ex_down: callable, Ey_up: callable, Ey_down: callable, 
                 curv_xx: callable, curv_xy: callable, curv_yy: callable) -> None:
        """Class that sets up the equations of motion in matrix form and solves them.

        Args:
            f0 (float): Resonance frequency of the RF-mode that primarily couples to the electron mode.
            Z0 (float): Impedance of the RF-mode.
            Ex (callable): Electric field in the x-direction. This function is inherited from the FullModel class.
            Ey (callable): Electric field in the y-direction. This function is inherited from the FullModel class.
            curv_xx (callable): Second derivative of the electrostatic potential:  d^2 / dx^2 V. This function is inherited from the PositionSolver class.
            curv_xy (callable): Second derivative of the electrostatic potential:  d^2 / dx dy V. This function is inherited from the PositionSolver class.
            curv_yy (callable): Second derivative of the electrostatic potential:  d^2 / dy^2 V. This function is inherited from the PositionSolver class.
        """
        self.f0 = f0
        self.Z0 = Z0
            
        # Electric field functions for the simple single-mode LC circuit    
        self.Ex = Ex
        self.Ey = Ey
        
        # Electric field functions (callables) for the coupled LC approach
        self.Ex_up = Ex_up
        self.Ex_down = Ex_down
        self.Ey_up = Ey_up
        self.Ey_down = Ey_down
        
        self.curv_xx = curv_xx 
        self.curv_xy = curv_xy
        self.curv_yy = curv_yy
    
    def setup_eom_coupled_lc(self, ri: ArrayLike, resonator_dict: Dict) -> tuple[ArrayLike]:
        """
        Set up the Matrix used for determining the electron frequency.
        :param electron_positions: Electron positions, in the form [x0, y0, x1, y1, ...]
        :return: M^(-1) * K
        """
        # omega0 = 2 * np.pi * self.f0
        # L = self.Z0 / omega0
        # C = 1 / (omega0**2 * L)
        C1 = resonator_dict['C1']
        C2 = resonator_dict['C2']
        Cdot = resonator_dict['Cdot']
        L1 = resonator_dict['L1']
        L2 = resonator_dict['L2']
        
        self.num_cavity_modes = 2
        
        # Identify the common and differential modes
        D = C1 * C2 + C1 * Cdot + C2 * Cdot

        M = np.array([[L1, 0], 
                      [0, L2]])

        K = np.array([[(C2 + Cdot) / D, Cdot / D], 
                      [Cdot / D, (C1 + Cdot) / D]])

        eigenvalues, _ = scipy.linalg.eigh(K, b=M)
        f0, f1 = np.sqrt(eigenvalues) / (2 * np.pi)
        
        # The differential mode is the smaller, because the coupling capacitance adds to the resonance
        self.f0_diff = np.min([f0, f1])
        # The common mode is higher, because the coupling capacitance doesn't participate in the resonance.
        self.f0_comm = np.max([f0, f1])
        
        if resonator_dict['mode'] == 'comm':
            self.f0 = self.f0_comm
        elif resonator_dict['mode'] == 'diff':
            self.f0 = self.f0_diff
        else:
            print("'mode' property was not understood. Please specify either 'comm' or 'diff'.")
            
        num_electrons = int(len(ri) / 2)
        xe, ye = r2xy(ri)

        # Set up the inverse of the mass matrix first
        # M = np.diag(np.array([C1 + Cdot] + [C2 + Cdot] + [m_e] * (2 * num_electrons)))
        # M[0, 1] = -Cdot
        # M[1, 0] = -Cdot
        
        M = np.diag(np.array([L1] + [L2] + [m_e] * (2 * num_electrons)))

        # Set up the kinetic matrix next
        Kij_plus, Kij_minus, Lij = np.zeros(np.shape(M)), np.zeros(np.shape(M)), np.zeros(np.shape(M))
        K = np.zeros((2 * num_electrons + 2, 2 * num_electrons + 2))
        
        # Row 1 and column 1 only have bare cavity information, and cavity-electron terms
        # K[0, 0] = 1 / L1
        # K[1, 1] = 1 / L2
        
        # K[2:num_electrons+2, 0] = K[0, 2:num_electrons+2] = q_e * self.Ex_up(xe, ye)
        # K[2:num_electrons+2, 1] = K[1, 2:num_electrons+2] = q_e * self.Ex_down(xe, ye)
        
        # K[num_electrons+2:2*num_electrons+2, 0] = K[0, num_electrons+2:2*num_electrons+2] = q_e * self.Ey_up(xe, ye)
        # K[num_electrons+2:2*num_electrons+2, 1] = K[1, num_electrons+2:2*num_electrons+2] = q_e * self.Ey_down(xe, ye)
        
        K[:2, :2] = np.array([[(C2 + Cdot) / D, Cdot / D], 
                              [Cdot / D, (C1 + Cdot) / D]])
        
        K[2:num_electrons+2, 0] = K[0, 2:num_electrons+2] = q_e / D * ( (C2 + Cdot) * self.Ex_up(xe, ye) - Cdot * self.Ex_down(xe, ye) )
        K[2:num_electrons+2, 1] = K[1, 2:num_electrons+2] = q_e / D * ( (C1 + Cdot) * self.Ex_down(xe, ye) - Cdot * self.Ex_up(xe, ye) )
        
        K[num_electrons+2:2*num_electrons+2, 0] = K[0, num_electrons+2:2*num_electrons+2] = q_e / D * ( (C2 + Cdot) * self.Ey_up(xe, ye) - Cdot * self.Ey_down(xe, ye) )
        K[num_electrons+2:2*num_electrons+2, 1] = K[1, num_electrons+2:2*num_electrons+2] = q_e / D * ( (C1 + Cdot) * self.Ey_down(xe, ye) - Cdot * self.Ey_up(xe, ye) )

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

        K[2:num_electrons+2, 2:num_electrons+2] = Kij_plus
        K[num_electrons+2:2*num_electrons+2, num_electrons+2:2*num_electrons+2] = Kij_minus
        K[2:num_electrons+2, num_electrons+2:2*num_electrons+2] = Lij
        K[num_electrons+2:2*num_electrons+2, 2:num_electrons+2] = Lij

        return K, M
    
    def setup_eom(self, ri: ArrayLike) -> tuple[ArrayLike]:
        """
        Set up the Matrix used for determining the electron frequency.
        :param electron_positions: Electron positions, in the form [x0, y0, x1, y1, ...]
        :return: M^(-1) * K
        """
        omega0 = 2 * np.pi * self.f0
        L = self.Z0 / omega0
        C = 1 / (omega0**2 * L)
        self.num_cavity_modes = 1

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

    def solve_eom(self, LHS: ArrayLike, RHS: ArrayLike, sort_by_cavity_participation: bool=True, cavity_mode_index: int=0) -> tuple[ArrayLike]:
        """Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
        The order of eigenvalues, and order of the columns of EVecs is coupled. By default scipy sorts this from low eigenvalue to high eigenvalue, however, 
        by flagging sort_by_cavity_participation, this function will return the eigenvalues and vectors sorted by largest cavity contribution first.

        Args:
            LHS (ArrayLike): K, analog of the spring constant matrix.
            RHS (ArrayLike): M, analog of the mass matrix.
            sort_by_cavity_participation (bool, optional): Sorts the eigenvalues/vectors by the participation in the first element of the eigenvector. Defaults to True.

        Returns:
            tuple[ArrayLike]: Eigenfrequencies, Eigenvectors
        """

        # EVals, EVecs = np.linalg.eig(np.dot(np.linalg.inv(RHS), LHS))
        EVals, EVecs = scipy.linalg.eigh(LHS, b=RHS)
        
        if sort_by_cavity_participation:
            # The cavity participation is the first element of each eigenvector, because that's how the matrix was constructed.
            cavity_participation = EVecs[cavity_mode_index, :]
            # Sort by largest cavity participation (argsort will normally put the smallest first, so invert it)
            sorted_order = np.argsort(np.abs(cavity_participation))[::-1]
            # Only the columns are ordered, the rows (electrons) are not shuffled. Keep the Evals and Evecs order consistent.
            EVecs = EVecs[:, sorted_order]
            EVals = EVals[sorted_order]
        
        return np.sqrt(EVals) / (2 * np.pi), EVecs
    
    def get_cavity_frequency_shift(self, LHS: ArrayLike, RHS: ArrayLike, cavity_mode_index: int=0) -> float:
        """Solves the equations of motion and calculates how to resonator frequency is affected.

        Args:
            LHS (ArrayLike): K, analog of the spring constant matrix.
            RHS (ArrayLike): M, analog of the mass matrix.

        Returns:
            float: Resonance frequency shift
        """
        
        eigenfrequencies, _ = self.solve_eom(LHS, RHS, sort_by_cavity_participation=True, cavity_mode_index=cavity_mode_index)
        return eigenfrequencies[0] - self.f0
    
    def plot_eigenvector(self, electron_positions: ArrayLike, eigenvector: ArrayLike, length: float=0.5, color: str='k') -> None:
        """Plots the eigenvector at the electron positions.

        Args:
            electron_positions (ArrayLike): Electron position array in length 2 * n_electrons. The order should be [x0, y0, x1, y1, ...]
            eigenvector (ArrayLike): Eigenvector to be plotted. Length should be 2 * n_electrons + 1, as a column output by solve_eom.
            length (float, optional): Length of the eigenvector in units of microns. Defaults to 0.5.
            color (str, optional): Face color of the arrow. Defaults to 'k'.
        """
        e_x, e_y = r2xy(electron_positions)
        N_e = len(e_x)

        # The first index of the eigenvector contains the charge displacement, thus we look at the second index and beyond.
        # Normalize the vector to 'length'
        evec_norm = eigenvector[self.num_cavity_modes:] / np.linalg.norm(eigenvector[self.num_cavity_modes:])
        # The x and y components are ordered differently than electron positions. This depends on the ordering of the K and M matrix, see setup_eom.
        dxs = (evec_norm * length)[:N_e]
        dys = (evec_norm * length)[N_e:]

        for e_idx in range(len(e_x)):
            width=0.025
            plt.arrow(e_x[e_idx] * 1e6, e_y[e_idx] * 1e6, dx=dxs[e_idx], dy=dys[e_idx], width=width, head_length=1.5*3 *width, head_width=3.5*width, 
                    edgecolor='k', lw=0.4, facecolor=color)
    
    def animate_eigenvectors(self, fig, axs_list: list, eigenvector_list: List[ArrayLike], electron_positions: ArrayLike, 
                             amplitude: float=0.5e-6, time_points: int=31, frame_interval_ms: int=10):
        """Make a matplotlib animation object for saving as a gif, or for displaying in a notebook.
        For use in displaying only:

        from IPython import display 
        ani = animate_eigenvectors(fig, axs, evecs.T, res['x'], amplitude=0.10e-6, time_points=21, frame_interval_ms=25)
        # Display animation
        video = ani.to_html5_video() 
        html = display.HTML(video) 
        display.display(html)
        
        # Save animation
        writer = animation.PillowWriter(fps=40, bitrate=1800)
        ani.save(savepath, writer=writer)

        Args:
            fig (matplotlib.pyplot.figure): Matplotlib figure handle.
            axs_list (matplotlib.pyplot.axes): List of axes, e.g. for subplots. 
            eigenvector_list (List[ArrayLike]): Eigenvector array. eigenvector_list[0] will be plot on axs_list[0] etc.
            electron_positions (ArrayLike): Electron coordinates in the format [x0, y0, x1, y1, ...]
            amplitude (float, optional): Amplitude of the motion in units of meters. Defaults to 0.5e-6.
            time_points (int, optional): Number of frames for one cycle (oscillation period). Defaults to 31.
            frame_interval_ms (int, optional): Interval between frames in milliseconds. Defaults to 10.

        Returns:
            _type_: matplotlib.animation.FuncAnimation object.
        """
        e_x, e_y = r2xy(electron_positions)
        N_e = len(e_x)

        all_points = list()
        for ax in axs_list:
            pts_data = ax.plot(e_x*1e6, e_y*1e6, 'ok', mfc='mediumseagreen', mew=0.5, ms=10, path_effects=[pe.SimplePatchShadow(), pe.Normal()])
            all_points.append(pts_data)

        # Only things in the update function will get updated.
        def update(frame):
            # Update the electron positions (green dots)
            for points, eigenvector in zip(all_points, eigenvector_list):
                evec_norm = eigenvector[self.num_cavity_modes:] / np.linalg.norm(eigenvector[self.num_cavity_modes:])
                dxs = (evec_norm * amplitude)[:N_e]
                dys = (evec_norm * amplitude)[N_e:]
                
                points[0].set_xdata((e_x + dxs * np.sin(2 * np.pi * frame / time_points)) * 1e6)
                points[0].set_ydata((e_y + dys * np.sin(2 * np.pi * frame / time_points)) * 1e6)

            return all_points, 

        # The interval is in milliseconds
        return animation.FuncAnimation(fig=fig, func=update, frames=time_points, interval=frame_interval_ms, repeat=True)
    
    def show_animation(self, matplotlib_animation) -> display.display:
        """Display an animation in a jupyter notebook. 

        Args:
            matplotlib_animation (matplotlib.animation.FuncAnimation): animation object, for example from `animate_eigenvectors`

        Returns:
            display.display: looped animation in html format.
        """
        # converting to an html5 video 
        video = matplotlib_animation.to_html5_video() 
        
        # embedding for the video 
        html = display.HTML(video) 
        
        # draw the animation 
        return display.display(html)
        
    def save_animation(self, matplotlib_animation, filepath) -> None:
        """Save a matplotlib animation to a gif format

        Args:
            matplotlib_animation (matplotlib.animation.FuncAnimation): animation object, for example from `animate_eigenvectors`
            filepath (str): filepath string that ends in .gif
        """
        writer = animation.PillowWriter(fps=40, bitrate=1800)

        matplotlib_animation.save(filepath, writer=writer)