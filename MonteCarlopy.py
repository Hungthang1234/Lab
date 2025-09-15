# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:15:26 2024

@author: ADMIN
"""

import numpy as np

# Define the function f(x, y, z)
def f(x, y, z):
    return 4 * x**3 + x * y**2 + 5 * y + y * z + 6 * z

# Monte Carlo integration function
def monte_carlo_integration(num_samples):
    # Generate random samples for x, y, z within the given bounds
    x_samples = np.random.uniform(0, 4, num_samples)
    y_samples = np.random.uniform(0, 3, num_samples)
    z_samples = np.random.uniform(0, 2, num_samples)

    # Evaluate the function at each point
    f_values = f(x_samples, y_samples, z_samples)

    # Compute the mean of f(x, y, z) and multiply by the volume of the domain
    volume = (4 - 0) * (3 - 0) * (2 - 0)  # Volume of the domain
    integral = np.mean(f_values) * volume

    return integral

# Run Monte Carlo integration with sufficient samples for 5 digits of precision
num_samples = 100000  # Increase number of samples for better precision
integral_approx = monte_carlo_integration(num_samples)

print(f"Approximate integral: {integral_approx:.5f}")
