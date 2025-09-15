# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:12:38 2024

@author: ADMIN
"""

import numpy as np

# Define the function f(x) = 4/(1 + x^2)
def f(x):
    return 4 / (1 + x**2)

# Simpson's rule implementation
def simpsons_rule(a, b, n):
    h = (b - a) / n  # Step size
    x = np.linspace(a, b, n+1)  # Generate n+1 points
    fx = f(x)  # Evaluate f(x) at each point
    
    # Apply Simpson's rule
    integral = h / 3 * (fx[0] + fx[-1] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-2:2]))
    return integral

# Parameters
a = 0
b = 1
n = 100000

# Calculate pi using Simpson's rule
pi_approx = simpsons_rule(a, b, n)
print("Approximate value of pi:", pi_approx)
