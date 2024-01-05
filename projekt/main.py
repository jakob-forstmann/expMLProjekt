from sklearn.model_selection import cross_validate
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_valence_range
from baselines import Estimator
from models import Default_Decision_Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
import argparse

def load_dataset():
    spotify_songs = create_dataset()
    return split_data(spotify_songs)

def evaluate(model:Estimator,X_train,y_train):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,scoring=evaluation_metrics,cv=5,return_train_score=True)
    print(f"Evaulating with model {str(model)}")
    print("Training Dataset:")
    print(f"Model{model}: root mean squarred error",cv_score["train_neg_root_mean_squared_error"])
    print("mean across 5 folds",cv_score["train_neg_root_mean_squared_error"].mean())
    print(f"Model{model}: mean absolute error",cv_score["train_neg_mean_absolute_error"])
    print("mean across 5 folds",cv_score["train_neg_mean_absolute_error"].mean())
    print("Test Dataset:")
    print(f"Model{model}: root mean squarred error",cv_score["test_neg_root_mean_squared_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())
    print(f"Model{model}: mean absolute error",cv_score["test_neg_mean_absolute_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="evaluate with different algorithms")
    possible_algorithms = Estimator.get_estimators()
    parser.add_argument("--model",choices=possible_algorithms)
    model_name = parser.parse_args().__dict__["model"]
    model = Estimator.possible_estimators[model_name]().model
    spotify_songs = create_dataset()
    x_train,x_test,y_train,y_test = split_data(spotify_songs)   
    #save_splitted_data([train,test],["spotify_dataset_train","spotify_dataset_test"])
    evaluate(model,x_train,y_train)
    