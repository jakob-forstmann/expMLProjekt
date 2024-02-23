from models.baselines import Estimator
from preprocessing import build_model
from experiments.find_DT_parameters import get_optimized_dt
from experiments.find_RDF_parameters import get_optimized_rdf
from experiments.find_lasso_parameters import get_optimized_linear_model
from experiments.find_MLP_parameters import get_optimized_mlp

class DecisionTreeWrapper(metaclass=Estimator):
    """ a class Wrapper that returns the optimized DT
        used to for the CLI"""
    def __init__(self):
        self.model = build_model(get_optimized_dt())


class RandomForestWrapper(metaclass=Estimator):
    """ a class Wrapper that returns the optimized 
        Random Forest.used to for the CLI"""
    def __init__(self):
        self.model = build_model(get_optimized_rdf())


class LinearModelWrapper(metaclass=Estimator):
    """ a class Wrapper that returns the optimized 
        linear Model.used to for the CLI"""
    def __init__(self):
        self.model = build_model(get_optimized_linear_model())


class NeuronalNetworkWrapper(metaclass=Estimator):
    """ a class Wrapper that returns the optimized 
        MLP.used to for the CLI"""
    def __init__(self):
        self.model = build_model(get_optimized_mlp())