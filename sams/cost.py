# -*- coding: utf-8 -*-

"""COST FUNCTIONS

This module contains classes of different cost functions for optimization.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 13/03/2017

"""

import numpy as np
from matrix import nuclear_norm
from transform import cube2matrix
from plotting import plotCost


class costTest(object):
    """Test cost function class

    This class implements a simple l2 norm test

    Parameters
    ----------
    y : np.ndarray
        Input original data array
    operator : class
        Degredation operator class

    """

    def __init__(self, y, operator):

        self.y = y
        self.op = operator

    def get_cost(self, x):
        """Get cost function

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        x : np.ndarray
            Input optimised data array

        """

        return np.linalg.norm(self.y - self.op(x))


class costFunction(object):
    """Cost function class

    This class implements the cost function for deconvolution

    Parameters
    ----------
    y : np.ndarray
        Input original data array
    grad : class
        Gradient operator class
    wavelet : class, optional
        Wavelet operator class ("sparse" mode only)
    weights : np.ndarray, optional
        Array of wavelet thresholding weights ("sparse" mode only)
    lambda_reg : float, optional
        Low-rank regularization parameter ("lowr" mode only)
    mode : str {'lowr', 'sparse'}, optional
        Deconvolution mode (default is "lowr")
    positivity : bool, optional
        Option to test positivity contraint (defult is "True")
    tolerance : float, optional
        Tolerance threshold for convergence (default is "1e-4")
    window : int, optional
        Iteration interval to test for convergence (default is "5")
    print_cost : bool, optional
        Option to print cost function value at each iteration (default is
        "True")
    residual : bool, optional
        Option to calculate the residual (default is
        "False")
    output : str, optional
        Output file name for cost function plot

    """

    def __init__(self, y, grad, wavelet=None, weights=None,
                 lambda_reg=None, mode='lowr',
                 positivity=True, tolerance=1e-4, window=1, print_cost=True,
                 residual=False, output=None):

        self.y = y
        self.grad = grad
        self.wavelet = wavelet
        self.lambda_reg = lambda_reg
        self.mode = mode
        self.positivity = positivity
        self.update_weights(weights)
        self.cost = 1e6
        self.cost_list = []
        self.tolerance = tolerance
        self.print_cost = print_cost
        self.residual = residual
        self.iteration = 1
        self.output = output
        self.window = window
        self.test_list = []

    def update_weights(self, weights):
        """Update weights

        Update the values of the wavelet threshold weights ("sparse" mode only)

        Parameters
        ----------
        weights : np.ndarray
            Array of wavelet thresholding weights

        """

        self.weights = weights

    def l2norm(self, x):
        """Calculate l2 norm

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l2 norm value

        """

        l2_norm = np.linalg.norm(self.y - self.grad.MX(x))

        if self.print_cost:
            print ' - L2 NORM:', l2_norm

        return l2_norm

    def l1norm(self, x):
        """Calculate l1 norm

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l1 norm value

        """

        x = self.weights * self.wavelet.op(x)

        l1_norm = np.sum(np.abs(x))

        if self.print_cost:
            print ' - L1 NORM:', l1_norm

        return l1_norm

    def nucnorm(self, x):
        """Calculate nuclear norm

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float nuclear norm value

        """

        x_prime = cube2matrix(x)

        nuc_norm = nuclear_norm(x_prime)

        if self.print_cost:
            print ' - NUCLEAR NORM:', nuc_norm

        return nuc_norm

    def check_cost(self, x):
        """Check cost function

        This method tests the cost function for convergence in the specified
        interval of iterations

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        bool result of the convergence test

        """

        if self.iteration % (4 * self.window):

            self.test_list.append(self.cost)

            return False

        else:

            self.test_list.append(self.cost)

            # a = (self.test_list[-2] - self.test_list[-1]) / self.window
            # b = np.abs(np.gradient(self.test_list[-2:], self.window)[-1])

            t1 = np.average(self.test_list[-4:-2], axis=0)
            t2 = np.average(self.test_list[-2:], axis=0)
            self.test_list = []

            test = (np.linalg.norm(t1 - t2) / np.linalg.norm(t1))

            if self.print_cost:
                print ' - CONVERGENCE TEST:', test
                print ''

            return test <= self.tolerance

    def check_residual(self, x):
        """Check residual

        This method calculates the residual between the deconvolution and the
        observed data

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        """

        self.res = np.std(self.y - self.grad.op(x)) / np.linalg.norm(self.y)

        if self.print_cost:
            print ' - STD RESIDUAL:', self.res

    def get_cost(self, x):
        """Get cost function

        This method calculates the full cost function and checks the result for
        convergence

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        bool result of the convergence test

        """

        if self.iteration % self.window:

            test = False

        else:

            if self.print_cost:
                print ' - ITERATION:', self.iteration

            self.cost_old = self.cost

            if self.residual:
                self.check_residual(x)

            if self.positivity and self.print_cost:
                print ' - MIN(X):', np.min(x)

            if self.mode == 'all':
                self.cost = (0.5 * self.l2norm(x) ** 2 + self.l1norm(x) +
                             self.nucnorm(x))

            elif self.mode == 'sparse':
                self.cost = 0.5 * self.l2norm(x) ** 2 + self.l1norm(x)

            elif self.mode == 'lowr':
                self.cost = (0.5 * self.l2norm(x) ** 2 + self.lambda_reg *
                             self.nucnorm(x))

            elif self.mode == 'grad':
                self.cost = 0.5 * self.l2norm(x) ** 2

            self.cost_list.append(self.cost)

            if self.print_cost:
                print ' - Log10 COST:', np.log10(self.cost)
                print ''

            test = self.check_cost(x)

        self.iteration += 1

        return test

    def plot_cost(self):
        """Plot cost function

        This method plots the cost function as function of iteration number

        """

        plotCost(self.cost_list, self.output)
