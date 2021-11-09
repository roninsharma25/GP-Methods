import pandas as pd
import numpy as np
import torch
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel, PeriodicKernel, PolynomialKernel, RQKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var, max_cg_iterations
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available as cuda_available, empty_cache
#from SALib.sample import saltelli
#from SALib.analyze import sobol
from os.path import join
from copy import deepcopy
import warnings
from similarity import *
warnings.filterwarnings("ignore")


class CompoundKernel(Kernel):
    def __init__(self, kernel_1, kernel_2, operation):
        super(CompoundKernel, self).__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.operation = operation

    def forward(self, x1, x2):
        if self.operation == 'add':
            output = self.kernel_1(x1, x2) + self.kernel_2(x1, x2)
        elif self.operation == 'mul':
            output = self.kernel_1(x1, x2) * self.kernel_2(x1, x2)
        
        return ScaleKernel(output)

class GPRegressionModel(ExactGP):
    """
    The gpytorch model underlying the TA_GP class.
    """
    def __init__(self, train_x, train_y, likelihood, kernel="matern", nu=0.5, power=4, index = 5, compound = False, k1 = None, k2 = None):
        """
        Constructor that creates objects necessary for evaluating GP.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # The Mattern Kernal is particularly well suited to models with abrupt transitions between success and failure.
        if compound:
            self.covar_module(CompoundKernel(k1, k2, 'add'))
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


class TA_GP():
    """
    Class for the GP model.
    """
    def __init__(self, training_data, output_col=6, kernel="matern", nu=0.5, thres = 0.01, ranges = [1, 1, 1, 1, 1], output_file = "output.csv", compound = False, k1 = None, k2 = None):
        if output_col <= 6: # 5 or 6
            self.output_type = "Gens to 99 percent"
        elif output_col <= 8: # 7 or 8
            self.output_type = "Rate after 100 gens"
        else:
            raise("The only model output columns this GP can be trained on are 5/6 and 7/8.")
        
        data = pd.read_csv(join("data", training_data), error_bad_lines = False)
        data = data.to_numpy(dtype="float")
        self.index = 4
        
        # The model input parameters in the training set.
        self.train_x = torch.from_numpy(data[:,:self.index]).float().contiguous() # 4 to 5

        # The selected output to train on. The standard error of each output is in the next column of the csv.
        self.train_y = torch.from_numpy(data[:,output_col-1:output_col]).float().contiguous().flatten()
        self.y_noise = torch.from_numpy(data[:,output_col:output_col+1]).float().contiguous().flatten()

        # Run model on the GPU using CUDA if it is available.
        if cuda_available():
            self.y_noise = self.y_noise.cuda()
        self.likelihood = FixedNoiseGaussianLikelihood(self.y_noise, learn_additional_noise=False)
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood, kernel=kernel, nu=nu, index=self.index, compound = False, k1 = None, k2 = None)

        if cuda_available():
            self.train_x, self.train_y, self.likelihood, self.model = self.train_x.cuda(), self.train_y.cuda(), self.likelihood.cuda(), self.model.cuda()

        # Some dictionaries for plotting and sensitivity analyses.
        self.default_params = {
                'Drive fitness': 0.99,
                'Release percent': 0.25,
                'Embryo cut rate': 0.95,
                'Germline cut rate': 0.99,
                'Random' : 0.5,
        }
        self.param_ranges = {
                "Drive fitness" : (0, 1),
                "Release percent" : (0, 1),
                "Embryo cut rate" : (0, 1),
                "Germline cut rate" : (0, 1),
                "Random" : (0, 1),
        }
        self.sa_params_dict = {
                "num_vars": self.index,
                "names": [k for k, v in self.param_ranges.items()][:self.index],
                "bounds": [v for k, v in self.param_ranges.items()][:self.index]
        }

    def save(self, filename, cpu=False):
        """
        Saves the trained GP model.
        """
        filename = join("models", filename)
        model_to_save = self.model
        if cpu:
            model_to_save = self.model.cpu()
        torch.save(model_to_save.state_dict(), f"{filename}.pth")
        if cpu:
            if cuda_available():
                self.model.cuda()
        print("Model saved.")

    def load(self, filename):
        """
        Loads a pre-trained GP model.
        """
        filename = join("models", filename)
        try:
            self.model.load_state_dict(torch.load(f"{filename}.pth"))
        except FileNotFoundError:
            try:
                self.model.load_state_dict(torch.load(filename))
            except FileNotFoundError:
                raise FileNotFoundError(f"{filename} not found.")
        # Torch requires that a loaded model be "retrained" on the training data.
        self.train(1)
        print("Model loaded.")

    def train(self, num_iterations):
        """
        Train the model.
        """
        # Set the model to training mode.
        self.model.train()
        self.likelihood.train()
        # Using the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)
        # "Loss" for GPs: the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        if num_iterations >= 100:
            print(f"Training for {num_iterations} iterations:")
        # Training loop:
        with max_cg_iterations(32000):
            for i in range(num_iterations):
                # Zero gradients from previous iteration:
                optimizer.zero_grad()
                # Output from model:
                output = self.model(self.train_x)
                # Calc loss and backprop gradients:
                loss = -mll(output, self.train_y)
                loss.backward()
                if (i+1) % 100 == 0:
                    print(f"Iter {i + 1}/{num_iterations} - Loss: {loss.item()}")
                optimizer.step()
                if cuda_available():
                    empty_cache()
        # Set the model to evaluation mode.
        self.model.eval()
        self.likelihood.eval()

    def predict(self, data):
        """
        Predicts y values, lower, and upper confidence for a data set.
        Takes data of the form of a panda dataframe.
        Just an alternative interface for predict_ys() really.
        """
        data = data.to_numpy(dtype="float")
        x = torch.from_numpy(data[:,:self.index]).float().contiguous() # 4 to 5
        if cuda_available():
            x = x.cuda()
        return self.predict_ys(x)

    def predict_ys(self, parsed_data):
        """
        Predicts y values from X values.
        Takes parsed data as a contiguous (cuda if available) torch tensor.
        """
        loader = DataLoader(TensorDataset(parsed_data), batch_size=1024, shuffle=False)
        mean, lower, upper = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
        with torch.no_grad(), fast_pred_var():
            for batch in loader:
                observed_pred = self.likelihood(self.model(batch[0]))
                cur_mean = observed_pred.mean
                if cuda_available():
                    mean = torch.cat([mean, cur_mean.cpu()])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower.cpu()])
                    upper = torch.cat([upper, cur_upper.cpu()])
                else:
                    mean = torch.cat([mean, cur_mean])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower])
                    upper = torch.cat([upper, cur_upper])
        return mean[1:], lower[1:], upper[1:]

    def sensitivity_analysis(self, base_sample=10000, param_ranges=None, verbose=False):
        """
        Perform a sensitivity analysis. Print the analysis if verbose=True.
        Returns a list of 3 pandas dataframes, where
        the first entry in the list is total effects, the second entry is first order, and the third entry is second order effects.
        """
        sa_params = deepcopy(self.sa_params_dict)
        if param_ranges:
            for key in param_ranges:
                if key not in self.default_params:
                    print(f"\"{key}\" not a valid parameter name. Ignoring.")
                if key in sa_params["names"]:
                    sa_params["bounds"][sa_params["names"].index(key)] = param_ranges[key]

        for i in range(len(sa_params["bounds"])):
            if type(sa_params["bounds"][i]) is not list and type(sa_params["bounds"][i]) is not tuple:
                sa_params["bounds"][i] = (sa_params["bounds"][i], sa_params["bounds"][i] + 0.00000001)

        # Generate samples
        param_values = saltelli.sample(sa_params, base_sample)
        # Evaluate the model at sampled points:
        x = np.zeros((len(param_values), self.index)) # 4 to 5
        for i in range(len(param_values)):
            x[i] = param_values[i]
        x = torch.from_numpy(x).float().contiguous()
        if cuda_available():
            x = x.cuda()
        y, _, _ = self.predict_ys(x)
        y = y.numpy()
        # Perform the sensitivity analysis:
        sa = sobol.analyze(sa_params, y, print_to_console=verbose)
        sa_df = sa.to_df()
        sa_df[0].columns = [c.replace('ST', 'Total Effects') for c in sa_df[0].columns]
        sa_df[1].columns = [c.replace('S1', 'First Order') for c in sa_df[1].columns]
        sa_df[2].columns = [c.replace('S2', 'Second Order') for c in sa_df[2].columns]
        sa_df.append(f"TARE sensitivity analysis ({self.output_type})")
        return sa_df

    def get_adaptive_points(self, num_points=100, output_filename="adaptive_run.sh", stochastic=False, n = 4):
        """
        Generates a new adaptively sampled data set.
        """
        points_to_test = "adaptive_test_set.csv"
        search_points = pd.read_csv(points_to_test)
        dataset = search_points.to_numpy(dtype="float")[:,:self.index] # 4 to 5
        mean, lower, upper = self.predict(search_points)
        mean, lower, upper = mean.numpy(), lower.numpy(), upper.numpy()
        thres = n ** 0.5 * 0.01 # n ** 0.5 * 1%

        selected_points = np.array([])
        while (len(selected_points) < num_points):
            value_of_points = upper - lower
            print('Number of New Points:', len(selected_points))
            input_1 = selected_points[0] if len(selected_points) else selected_points

            if stochastic:
                value_of_points /= value_of_points.sum()
                value_of_points[-1] = value_of_points[-1] + (1 - value_of_points.sum())
                input_2 = np.random.choice(len(value_of_points), num_points,  replace=False, p=value_of_points)
            else:
                # Non-random point selection: the list is sorted by 95%CI width, and the top "num_points" points are selected.
                # This method of selecting additional points may be innapropriate
                # depending on the nature of the list of candadate points!
                input_2 = np.argpartition(value_of_points, -num_points)[-num_points:]
         
            selected_points = np.concatenate((input_1, input_2)).astype(int)
            points = [dataset[int(i)].tolist() for i in selected_points]
            selected_points = detectSimilarPoints(self.train_x, points, [1, 1, 1, 1], thres)

        # Restrict to first set of points if more than the max
        points = selected_points[:num_points]

        # The points are output in an sh script to locally run the next set of simulations.
        max_simultaneous_procs = 10
        run_number = 1
        print(f"Outputting adaptive paramset to \"{output_filename}\".")
        with open(output_filename, 'w+', newline='\n') as f:
            for p in points:
                f.write(f"python3 TA_systems_driver.py --DRIVE_FITNESS_VALUE {p[0]} --DROP_RATE {p[1]} --EMBRYO_RESISTANCE_RATE {p[2]} --GERMLINE_RESISTANCE_RATE {p[3]} > data/{run_number}.part &")
                f.write('\n')
                if run_number % max_simultaneous_procs == 0:
                    f.write("wait")
                    f.write('\n')
                    f.write("now=$(date +\"%T\")")
                    f.write('\n')
                    f.write(f"echo Done with {run_number} of {num_points} simulations at $now.")
                    f.write('\n')
                run_number += 1
            f.write("wait\ncd data\ncat *.part > adaptive.csv\nrm *.part")
            f.write('\n')
        print("Done.")
