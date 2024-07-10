import time
from quantum_electron import FullModel
import numpy as np
from matplotlib import pyplot as plt
from alive_progress import alive_bar

def test_performance():
    """Tests the performance of the minimization using the FullModel class. 
    We construct a parabolic potential and minimize electron clusters of various sizes.
    """
    
    def analytical_potential(x, y):
        return (x ** 2 + y ** 2) / micron ** 2
    
    x = np.linspace(-10, 10, 401)
    y = np.linspace(-10, 10, 401)

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
    
    # Number of repetitions to calculate the errorbars.
    repetitions = 3
    # Number of electrons in the cluster.
    n_electrons = np.array([1, 2, 4, 6, 8, 10, 20, 40, 80, 100, 200, 400])
    sol_times = np.zeros([3, len(n_electrons)])

    # Loop over various electron cluster sizes. 
    with alive_bar(repetitions * len(n_electrons), force_tty=True) as bar:
        for r in range(repetitions):
            for k, n in enumerate(n_electrons):
                t0 = time.time()
                res = fm.get_electron_positions(n_electrons=n)
                t1 = time.time()

                # Save the solution time in an array
                sol_times[r, k] = t1 - t0

                assert res['success'], res["message"]
                bar()
        
    # Plot the results 
    fig = plt.figure(figsize=(6., 4.))
    plt.errorbar(n_electrons, np.mean(sol_times, axis=0), yerr=np.std(sol_times, axis=0), fmt='-o', capsize=5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Number of electrons")
    plt.ylabel("Solution time (s)")

    # Try to save the figure in the images folder. It can be used in the readme.
    # fig.savefig("images/performance.png", dpi=100, bbox_inches='tight', pad_inches=0.05)

    assert True