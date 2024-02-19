from baselines import Estimator
from experiments.find_DT_parameters import get_optimized_dt
from experiments.find_RDF_parameters import get_optimized_rdf
from experiments.find_lasso_parameters import get_optimized_linear_model
from preprocessing import build_model


class Decision_Tree_Wrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_dt())

    def __str__(self):
        return "decision tree"

class  Random_Forest_Regressor_Wrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_rdf)

    def __str__(self):
        return "Mean Baseline"

class Linear_Model_Wrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_linear_model())
