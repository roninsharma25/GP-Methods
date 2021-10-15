# https://emukit.github.io/experimental-design/

from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design import ExperimentalDesignLoop
import TA_gp

training_data = ""
batch_size = 10
max_iterations = 10

model = TA_gp(training_data)
model_emukit = GPyModelWrapper(model)
model_variance = ModelVariance(model = model_emukit)
target_function = 0 # UPDATE THIS

parameter_space = ParameterSpace([
    ContinuousParameter("--DRIVE_FITNESS_VALUE", 0, 1),
    ContinuousParameter("--DROP_RATE", 0, 1),
    ContinuousParameter("--EMBRYO_RESISTANCE_RATE", 0, 1),
    ContinuousParameter("--GERMLINE_RESISTANCE_RATE", 0, 1),
    ])

expdesign_loop = ExperimentalDesignLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = model_variance,
                                         batch_size = batch_size)

expdesign_loop.run_loop(target_function, max_iterations)