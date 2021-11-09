from TA_gp import *
from TA_plot import *
from similarity import *
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
import subprocess

NUM_ITERATIONS = 1
NUM_POINTS = 100

def singleKernels():
    for kernel in ['rbf', 'rq']:
        model = TA_GP('1.csv', kernel = kernel)
        model.train(NUM_ITERATIONS)
        model.save('{}_nosampling'.format(kernel), True)

        # Sampling
        model.get_adaptive_points(NUM_POINTS)
        subprocess.run(['sh', 'adaptive_run.sh'])
        subprocess.run(['cat', 'data/adaptive.csv', '>>', 'data/1.csv'])
        model.train(NUM_ITERATIONS)
        model.save('{}_sampling'.format(kernel), True)


def multipleKernels():
    model = TA_GP('1.csv', compound = 'add', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('addMaternRBF_nosampling', True)

    # Sampling
    model.get_adaptive_points(NUM_POINTS)
    subprocess.run(['sh', 'adaptive_run.sh'])
    subprocess.run(['cat', 'data/adaptive.csv', '>>', 'data/1.csv'])
    model.train(NUM_ITERATIONS)
    model.save('addMaternRBF_sampling', True)

    model = TA_GP('1.csv', compound = 'mul', k1 = MaternKernel(), k2 = RBFKernel())
    model.train(NUM_ITERATIONS)
    model.save('mulMaternRBF_nosampling', True)

    # Sampling
    model.get_adaptive_points(NUM_POINTS)
    subprocess.run(['sh', 'adaptive_run.sh'])
    subprocess.run(['cat', 'data/adaptive.csv', '>>', 'data/1.csv'])
    model.train(NUM_ITERATIONS)
    model.save('mulMaternRBF_sampling', True)


def testKernels():
    singleKernels()
    multipleKernels()


if __name__ == '__main__':
    testKernels()
