from sklearn.dummy import DummyRegressor
from dataset_statistics import calculate_most_frequent_value,count_unique_valence_values
from preprocessing import create_dataset
from numpy import random
from abc import abstractmethod

class Estimator(type):
    possible_estimators = {}
    def __new__(cls,name,base,namespaces):
        specific_estimator = super().__new__(cls,name,base,namespaces)
        class_name = name.replace("_"," ")
        cls.possible_estimators[class_name] = specific_estimator
        return specific_estimator

    def fit(self,x_train,x_test,y_train,y_test):
        return self.model.fit(x_train,x_test)

    @abstractmethod
    def __str__(self):
        pass
    

class Majority_Baseline(metaclass=Estimator):
    def __init__(self):
        spotify_songs = create_dataset()
        self.frequent_value = calculate_most_frequent_value(spotify_songs)
        self.model = DummyRegressor(strategy="constant",constant=self.frequent_value)

    def __str__(self):
        return f"Majority Baseline with most frequent valence{self.frequent_value}"


class  Mean_Baseline(metaclass=Estimator):
    def __init__(self):
        self.model = DummyRegressor(strategy="mean")

    def __str__(self):
        return "Mean Baseline"

class Random_Baseline(metaclass=Estimator):
    def __init__(self):
        spotify_songs = create_dataset()
        distribution = count_unique_valence_values(spotify_songs)
        distribution_probs = distribution.values/distribution.sum()
        self.random_valence = random.choice(distribution.index,p=distribution_probs)
        self.model =  DummyRegressor(strategy="constant",constant=self.random_valence)

    def __str__(self):
        return f"Random Baseline with random value {self.random_valence}"

