# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:13:48 2024

@author: ADMIN
"""

import numpy as np

# Define the function f(x) = 4/(1 + x^2)
def f(x):
    return 4 / (1 + x**2)

# Simpson's rule implementation with error tolerance
def simpsons_rule(a, b, tolerance):
    n = 2  # Start with 2 intervals
    integral_old = 0
    while True:
        h = (b - a) / n  # Step size
        x = np.linspace(a, b, n+1)  # Generate n+1 points
        fx = f(x)  # Evaluate f(x) at each point

        # Apply Simpson's rule
        integral_new = h / 3 * (fx[0] + fx[-1] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-2:2]))

        # Check for convergence
        if abs(integral_new - integral_old) < tolerance:
            break

        integral_old = integral_new
        n *= 2  # Double the number of intervals for better accuracy
    
    return integral_new

# Parameters
a = 0
b = 1
tolerance = 1e-5  # Error tolerance to achieve 5 decimal precision

# Calculate integral using Simpson's rule with error tolerance
integral_approx = simpsons_rule(a, b, tolerance)
print(f"Approximate integral: {integral_approx}")
