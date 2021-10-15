import pandas as pd
import numpy as np
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, PeriodicKernel, PolynomialKernel, RQKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

import warnings
warnings.filterwarnings("ignore")

class GPRegressionModel(ExactGP):
    """
    The gpytorch model underlying the TA_GP class.
    """
    def __init__(self, train_x, train_y, likelihood, kernel="matern", nu=0.5, power=4):
        """
        Constructor that creates objects necessary for evaluating GP.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # The Mattern Kernal is particularly well suited to models with abrupt transitions between success and failure.
        if kernel == "matern":
            self.covar_module = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=4))
        elif kernel == "rbf":
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel == "periodic":
            self.covar_module = ScaleKernel(PeriodicKernel())
        elif kernel == "polynomial":
            self.covar_module = ScaleKernel(PolynomialKernel(power=power))
        elif kernel == 'rq': # mixture of RBF kernels
            self.covar_module = ScaleKernel(RQKernel())
        else:
            raise Exception("Kernel must be from [matern, rbf, periodic, polynomial, rq].")

    def forward(self, x):
        """
        Takes in nxd data x and returns a MultivariateNormal with the prior mean and covariance evaluated at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)