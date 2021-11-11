from TA_gp import *
from TA_plot import *
from similarity import *
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
import subprocess

NUM_ITERATIONS = 5000
NUM_POINTS = 750

def singleKernels():
    for kernel in ['rbf', 'rq']:
        model = TA_GP('1.csv', kernel = kernel)
        model.train(NUM_ITERATIONS)
        model.save(f'{kernel}_nosampling', True)

        # Sampling
        model.get_adaptive_points(NUM_POINTS)
        #subprocess.run(['sh', 'adaptive_run.sh'])
        subprocess.call(f'cat data/1.csv data/adaptive.csv > data/{kernel}_sampling.csv', shell = True)
        model = TA_GP(f'{kernel}_sampling.csv', kernel = kernel)
        model.train(NUM_ITERATIONS)
        model.save(f'{kernel}_sampling', True)

def multipleKernels():
    model = TA_GP('1.csv', compound = 'add', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('addMaternRBF_nosampling', True)

    # Sampling
    model.get_adaptive_points(NUM_POINTS)
    subprocess.run(['sh', 'adaptive_run.sh'])
    subprocess.call('cat data/1.csv data/adaptive.csv > data/addMaternRBF_sampling.csv', shell = True)
    model = TA_GP('addMaternRBF_sampling.csv', compound = 'add', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('addMaternRBF_sampling', True)

    model = TA_GP('1.csv', compound = 'mul', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('mulMaternRBF_nosampling', True)

    # Sampling
    model.get_adaptive_points(NUM_POINTS)
    subprocess.run(['sh', 'adaptive_run.sh'])
    subprocess.call('cat data/1.csv data/adaptive.csv > data/mulMaternRBF_sampling.csv', shell = True)
    model = TA_GP('mulMaternRBF_sampling.csv', compound = 'mul', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('mulMaternRBF_sampling', True)

def testKernels():
    singleKernels()
    multipleKernels()

if __name__ == '__main__':
    testKernels()
