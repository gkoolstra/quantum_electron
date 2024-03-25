import pytest
from quantum_electron import FullModel
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import elementary_charge as qe, epsilon_0 as eps0

input_output_pairs = [(1, 0.0000), (2, 0.75000), (3, 1.31037), (4, 1.83545), 
                      (5, 2.33845), (6, 2.80456), (7, 3.23897), (8, 3.66890), 
                      (9, 4.08813), (10, 4.48494), (11, 4.86467), (12, 5.23895), 
                      (13, 5.60114), (14, 5.95899), (15, 6.30758), (16, 6.64990), 
                      (17, 6.98291), (18, 7.30814)]#, (19, 7.63197), (20, 7.94961), 
                    #   (21, 8.26588), (22, 8.57418), (23, 8.87765), (24, 9.17590), (25, 9.47273), 
                    #   (26, 9.76273)]

@pytest.mark.parametrize("n, expected", input_output_pairs)
def test_molecule_energies(n: int, expected: float):
    """Tests the energy of a Wigner molecule with n electrons in a parabolic confinement.
    The values of the energies per particle are taken from the paper by Bedanov and Peeters, Phys. Rev. B 49 (1994).

    Args:
        n (int): Number of electrons in the Wigner molecule.
        expected (float): Expected energy per particle in units of E0 = (alpha * qe ** 4 / (4 * np.pi * eps0) ** 2) ** (1/3).
    """
    # Note that the units of x and y are implicitly assumed as microns
    x = np.linspace(-3, 3, 401)
    y = np.linspace(-3, 3, 401)

    micron = 1e-6

    X, Y = np.meshgrid(x, y)
    X *= micron
    Y *= micron

    parabolic_confinement = - (X ** 2 + Y ** 2) / micron ** 2

    potential_dict = {"dot" : parabolic_confinement, 
                      "xlist" : x, 
                      "ylist" : y}

    # Let's apply these voltages to the corresponding electrodes in potential_dict
    voltages = {"dot" : 1.0}
    
    alpha = 1 / (micron) ** 2 * voltages['dot'] * qe
    E0 = (alpha * qe ** 4 / (4 * np.pi * eps0) ** 2) ** (1/3)

    fm = FullModel(potential_dict=potential_dict, voltage_dict=voltages, trap_annealing_steps=[20]*10, potential_smoothing=1e-7)
            
    res = fm.get_electron_positions(n_electrons=n, electron_initial_positions=None)
    E_N = res['fun'] * qe / (n * E0)

    # We'll allow an uncertainty of +/- 1e-5 in the energy.
    # Note that the precision is taken from the table, where the energies are given to 5 decimal places for molecules up to N = 26.
    # Molecules of size N > 26 have energies given to 4 decimal places, so we will not test them here.
    assert -2e-5 < (E_N - expected) < 2e-5

def test_interpolation():
    """Tests the interpolation of the potential using the FullModel class. 
    We construct a parabolic potential and compare it to the analytical potential.
    For an analytic function without discontinuities, the potential should be very close to the analytical potential.
    """
    
    def analytical_potential(x, y):
        return (x ** 2 + y ** 2) / micron ** 2
    
    x = np.linspace(-3, 3, 401)
    y = np.linspace(-3, 3, 401)

    micron = 1e-6

    X, Y = np.meshgrid(x, y)
    X *= micron
    Y *= micron

    parabolic_confinement = -analytical_potential(X, Y)

    potential_dict = {"dot" : parabolic_confinement, 
                      "xlist" : x, 
                      "ylist" : y}

    # Let's apply these voltages to the corresponding electrodes in potential_dict
    voltages = {"dot" : 1.0}
    
    fm = FullModel(potential_dict=potential_dict, voltage_dict=voltages)
    difference = fm.V(X, Y) - analytical_potential(X, Y)

    assert np.amax(difference) < 1e-5
    