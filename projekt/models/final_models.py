from models.baselines import Estimator
from preprocessing import build_model
from experiments.find_DT_parameters import get_optimized_dt
from experiments.find_RDF_parameters import get_optimized_rdf
from experiments.find_lasso_parameters import get_optimized_linear_model

class DecisionTreeWrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_dt())

    def __str__(self):
        return "decision tree"

class RandomForestWrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_rdf)

    def __str__(self):
        return "Mean Baseline"
class LinearModelWrapper(metaclass=Estimator):
    def __init__(self):
        self.model = build_model(get_optimized_linear_model())

