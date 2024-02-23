from sklearn.dummy import DummyRegressor
from dataset_statistics import calculate_most_frequent_value,count_unique_valence_values
from models.preprocessing import create_dataset
from numpy import random
from abc import abstractmethod


class Estimator(type):
    """ a Metaclass used to store and list the possible models"""
    possible_estimators = {}
    def __new__(mcs,name,base,namespaces):
        specific_estimator = super().__new__(mcs,name,base,namespaces)
        class_name = name.replace("Wrapper","")
        class_name = "".join(" " + char if char.isupper() else char for char in class_name).strip()
        mcs.possible_estimators[class_name] = specific_estimator
        return specific_estimator

    @classmethod
    def get_estimators(mcs):
        return mcs.possible_estimators

    @classmethod
    def get_model(mcs,model_name):
        """returns an Instance of the Class model_name
            or an Instance of the Majority Baseline 
            if no name is given"""
        if model_name is not None:
            return mcs.possible_estimators[model_name]().model
        return MajorityBaseline().model

    def fit(self,x_train,x_test,y_train,y_test):
        return self.model.fit(x_train,x_test)


class MajorityBaseline(metaclass=Estimator):
    """ a Baseline that always predicts the most frequent value"""
    def __init__(self):
        self.frequent_value = calculate_most_frequent_value()
        self.model = DummyRegressor(strategy="constant",constant=self.frequent_value)

  
class MeanBaseline(metaclass=Estimator):
    """ a Baseline that always predicts the mean of the dataset"""
    def __init__(self):
        self.model = DummyRegressor(strategy="mean")

   
class RandomBaseline(metaclass=Estimator):
    """ a Baseline that picks a random value from the dataset 
        and predict this value for every datapoint"""
    def __init__(self):
        distribution = count_unique_valence_values()
        distribution_probs = distribution.values/distribution.sum()
        self.random_valence = random.choice(distribution.index,p=distribution_probs)
        self.model =  DummyRegressor(strategy="constant",constant=self.random_valence)
