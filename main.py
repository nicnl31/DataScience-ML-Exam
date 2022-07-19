import numpy as np
import scipy.integrate as integrate
from numpy.linalg import *
from matplotlib import pyplot as plt


# PROBLEM DATA ---------------------------------------------------------------------------------------------------------
A = np.array([[1, 2], [0, 2]])

B = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])


def sigmoid(z):
    """
    Computes the sigmoid function of argument z.
    Args:
        z: float
            Argument of the sigmoid function.

    Returns:
        The sigmoid function of z.
    """
    return 1 / (1 + np.exp(-z))


def plot_sigmoid(start, stop):
    """
    Plots the sigmoid function with evenly spaced points .1 apart, from start to stop.
    """
    points = np.arange(start, stop + 1.1, .1)
    sigmoid_points = [sigmoid(i) for i in points]
    plt.plot(points, sigmoid_points)
    plt.title(f'Sigmoid function plot, x = {start} to x = {stop}')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.show()


# ANSWER ---------------------------------------------------------------------------------------------------------------
# Problem 4: #
def classify(data):
    """
    Classifies a single data point as 1 if sigmoid(data) >= 0.5, and 0 otherwise.
    """
    return 1 if sigmoid(data) >= 0.5 else 0


# Problem 5: #
def left_riemann_integral(start=-5, stop=5, num_intervals=1000000):
    """
    Computes the definite integral of the sigmoid function using the Left Riemann sum method.
    The higher the number of intervals, the more accurate the result.

    Returns the Left Riemann sum integral, and one that's from the Scipy package for comparison.
    """
    # Declare evenly spaced points in the interval
    riemann = np.linspace(start, stop, num_intervals+1)

    # Compute the integral
    base = (riemann[-1] - riemann[0]) / num_intervals
    integral = 0
    for i in riemann[:-1]:  # Skip the last point in the left Riemann sum
        integral += base * sigmoid(i)

    # Check error with Scipy's integral tool
    sy_integral = integrate.quad(lambda z: sigmoid(z), start, stop)
    error = abs(integral - sy_integral[0])
    print(f"Integral calculated with Left Riemann sum: {integral} \n"
          f"Integral calculated with the Scipy package: {sy_integral[0]} \n"
          f"Error: {error}")
    return None


def answer():
    """
    Answers for problem 1, 2, 3, 5.
    """
    # Problem 1: #
    print(f"Problem 1: \n"
          f"Eigenvalues: {eig(A)[0]} \n"
          f"Unit eigenvector matrix: \n {eig(A)[1]} \n"
          "----------------------")

    # Problem 2: #
    print(f"Problem 2: \n"
          f"Resulting column vector matrix: \n{A @ B} \n"
          "----------------------")

    # Problem 3: #
    plot_sigmoid(-5, 5)

    # Problem 5: #
    print(f"Problem 5: \n")
    left_riemann_integral()


answer()
