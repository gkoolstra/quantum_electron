import numpy as np
from .utils import xy2r, make_potential, find_minimum_location
from typing import Optional, Dict
from numpy.typing import ArrayLike

micron = 1e-6


class InitialCondition:
    """
    Class to generate initial conditions for a given potential energy landscape.

    Example usage:
    ic = InitialCondition(potential_dict, voltage_dict)
    init_cond = ic.make_by_chemical_potential(max_electrons, chemical_potential, min_spacing)

    f = FullModel(potential_dict, voltage_dict)
    f.get_electron_positions(n_electrons=len(init_cond) // 2, initial_condition=init_cond)
    """

    def __init__(self, potential_dict: Dict[str, ArrayLike], voltage_dict: Dict[str, ArrayLike]):
        """Initializes the InitialCondition class.

        Args:
            potential_dict (Dict[str, ArrayLike]): Dictionary containing at least the keys also present in the voltages dictionary.
            The 2d-array associated with each key contains the coupling coefficient for the respective electrode in space.
            voltage_dict (Dict[str, float]): Dictionary with electrode names as keys. The value associated with each key is the voltage
            applied to each electrode
        """
        self.potential_dict = potential_dict
        self.voltage_dict = voltage_dict

    def make_by_chemical_potential(self, max_electrons: int, chemical_potential: float, min_spacing: float = 0.1) -> ArrayLike:
        """Makes an initial condition for a given chemical potential. The initial condition is a set of random points with a minimum 
        spacing. The number of points is determined by the chemical potential and the potential energy landscape.
        The algorithm will try to fill the dot with electrons until it reaches the desired number of electrons: max_electrons.
        If there is no space for more electrons, the algorithm will return only the electrons that fit and print a warning message. 

        Args:
            max_electrons (int): The maximum number of electrons that will be attempted to fit into the designated area.
            chemical_potential (float): The chemical potential for the electrons. This will be used to determine the area to be filled.
            min_spacing (float, optional): Minimum spacing between electrons. Defaults to 0.1.

        Returns:
            ArrayLike: array of electron positions in the order np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        """
        z = -make_potential(self.potential_dict, self.voltage_dict)
        dot = (z < chemical_potential) * z
        bounds, dot_min, dot_max = self._dot_area(dot)
        points = self._generate_points(
            max_electrons, bounds, dot, dot_min, dot_max, epsilon=min_spacing) * micron
        init_condition = xy2r(points[:, 0], points[:, 1])

        return init_condition

    def make_circular(self, n_electrons: int, coor: Optional[tuple] = None, min_spacing: float = 0.1) -> ArrayLike:
        """Generates an array with electron coordinates in a circular pattern.

        Args:
            n_electrons (int): Number of electrons to be generated.
            coor (Optional[tuple], optional): Center of the circular pattern. Defaults to None.
            min_spacing (float, optional): Minimum spacing between electrons. Defaults to 0.1.

        Returns:
            ArrayLike: array of electron positions in the order np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        """
        if coor is None:
            coor = find_minimum_location(
                self.potential_dict, self.voltage_dict)

        radius = min_spacing * micron * n_electrons / (2 * np.pi)
        # Generate initial guess positions for the electrons in a circle with certain radius.
        init_trap_x = np.array([coor[0] * 1e-6 + radius * np.cos(2 *
                                                                 np.pi * n / float(n_electrons)) for n in range(n_electrons)])
        init_trap_y = np.array([coor[1] * 1e-6 + radius * np.sin(2 *
                                                                 np.pi * n / float(n_electrons)) for n in range(n_electrons)])

        init_condition = xy2r(np.array(init_trap_x), np.array(init_trap_y))

        return init_condition

    def make_rectangular(self, n_electrons: int, coor: tuple = (0, 0), dxdy: tuple = (2, 2), n_rows: int = 2) -> ArrayLike:
        """Generates an array with electron coordinates in a rectangular pattern.

        Args:
            n_electrons (int): Number of electrons to be generated.
            coor (tuple, optional): Center coordinate of the rectangle. Defaults to (0, 0).
            dxdy (tuple, optional): Width and height of the rectangle. Defaults to (2, 2).
            n_rows (int, optional): Number of rows. Defaults to 2.

        Returns:
            ArrayLike: array of electron positions in the order np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        """
        xmin = coor[0] - dxdy[0] / 2
        xmax = coor[0] + dxdy[0] / 2

        ymin = coor[1] - dxdy[1] / 2
        ymax = coor[1] + dxdy[1] / 2

        init_x = np.tile(np.linspace(
            xmin, xmax, n_electrons // n_rows), n_rows) * micron
        init_y = np.repeat(np.linspace(ymin, ymax, n_rows),
                           n_electrons // n_rows) * micron
        init_condition = xy2r(init_x, init_y)

        return init_condition

    def _no_overlap(self, existing_points: list, additional_point: tuple, epsilon: float) -> bool:
        """Helper function for make_by_chemical_potential.
        Checks if a new point overlaps with any of the existing points.

        Args:
            existing_points (list): List of existing points.
            additional_point (tuple): New point to be added.
            epsilon (float): Minimum spacing between electrons.

        Returns:
            bool: True if there is no overlap, False otherwise.
        """
        trial_points = existing_points.copy()
        trial_points.append(additional_point)
        x = [p[0] for p in trial_points]
        y = [p[1] for p in trial_points]
        X, Y = np.meshgrid(x, y)

        R = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
        np.fill_diagonal(R, 100)
        if np.min(np.min(R, axis=1)) < epsilon:
            return False
        else:
            return True

    def _dot_area(self, dot: ArrayLike) -> tuple:
        """Helper function for make_by_chemical_potential.
        Extracts the extent of the dot and the minimum and maximum values of the potential

        Args:
            dot (ArrayLike): Masked 2d array. The array has zeros outside the dot area.

        Returns:
            tuple: bounds, np.min(nonzero_dot), np.max(nonzero_dot)
        """
        # Returns the x-indices for which dot is non-zero
        x_slice = np.where(np.abs(np.mean(dot, axis=1)) > 0)[0]
        
        # Returns the y-indices for which dot is non-zero
        y_slice = np.where(np.abs(np.mean(dot, axis=0)) > 0)[0]
        
        x1 = self.potential_dict['xlist'][x_slice[0]]
        x2 = self.potential_dict['xlist'][x_slice[-1]]

        y1 = self.potential_dict['ylist'][y_slice[0]]
        y2 = self.potential_dict['ylist'][y_slice[-1]]

        bounds = [x1, x2, y1, y2]
        nonzero_dot = dot[dot != 0]
        return bounds, np.min(nonzero_dot), np.max(nonzero_dot)

    def _density_function(self, x: ArrayLike, y: ArrayLike, dot: ArrayLike, dot_min: float, dot_max: float) -> float:
        """Helper function for make_by_chemical_potential.

        Args:
            x (ArrayLike): 
            y (ArrayLike): 
            dot (ArrayLike): Masked 2d array with zeros outside the dot area.
            dot_min (float): Minimum value of the potential inside the dot area.
            dot_max (float): Maximum value of the potential inside the dot area.

        Returns:
            float: 
        """
        # Find the minimum and maximum x indices
        xFloor = np.argmax(self.potential_dict['xlist'] > x)-1
        xCeil = np.argmax(self.potential_dict['xlist'] > x)

        # Find the minimum and maximum y indices
        yFloor = np.argmax(self.potential_dict['ylist'] > y)-1
        yCeil = np.argmax(self.potential_dict['ylist'] > y)

        dx = self.potential_dict['xlist'][xCeil] - \
            self.potential_dict['xlist'][xFloor]
        dy = self.potential_dict['ylist'][yCeil] - \
            self.potential_dict['ylist'][yFloor]

        value_floor_left = (self.potential_dict['xlist'][xCeil] - x)/dx * dot[xFloor, yFloor] + \
            (x - self.potential_dict['xlist'][xFloor])/dx * dot[xCeil, yFloor]

        value_ceil_left = (self.potential_dict['xlist'][xCeil] - x)/dx * dot[xFloor, yCeil] + \
            (x - self.potential_dict['xlist'][xFloor])/dx * dot[xCeil, yCeil]

        interpolated_value = (self.potential_dict['ylist'][yCeil] - y)/dy * value_floor_left + \
            (y - self.potential_dict['ylist'][yFloor])/dy * value_ceil_left

        return (interpolated_value-dot_min)/(dot_max-dot_min)

    def _generate_points(self, max_electrons: int, bounds: list, dot: ArrayLike, dot_min: float, dot_max: float, epsilon: float, verbose: bool = True) -> ArrayLike:
        """Fills the dot with electrons until it reaches the desired number of electrons.
        The points are generated randomly and checked for overlap with the existing points.
        It will retry up to 100 times to add additional points that do not overlap.

        Args:
            max_electrons (int): Maximum number of electrons to be generated.
            bounds (list): Boundaries of the dot area.
            dot (ArrayLike): Masked 2d array. The array has zeros outside the dot area.
            dot_min (float): Minimum value of the potential inside the dot area.
            dot_max (float): Maximum value of the potential inside the dot area.
            epsilon (float): Minimum spacing between electrons.
            verbose (bool, optional): Prints a warning is not all electrons fit inside the dot. Defaults to True.

        Returns:
            ArrayLike: 2d array of points, the first column is the x coordinates, the second column the y coordinates.
        """
        points = []
        max_failures = 100
        failures = 0

        # Tries to add electrons to the dot until it reaches the desired number of electrons
        while len(points) < max_electrons and failures < max_failures:
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[2], bounds[3])

            # Add a random point if it is below the chemical potential and does not overlap with any other point
            if np.random.rand() < self._density_function(x, y, dot, dot_min, dot_max) and self._no_overlap(points, (x, y), epsilon=epsilon):
                points.append((x, y))
                failures = 0
            else:
                failures += 1

        if (failures == max_failures) and verbose:
            print(
                f'WARNING in creating initial condition: could not fit more than {len(points)} electrons.')

        return np.array(points)
