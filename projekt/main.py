from sklearn.model_selection import cross_validate
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_valence_range
from baselines import Estimator
import argparse

def load_dataset():
    spotify_songs = create_dataset()
    return split_data(spotify_songs)

def evaluate(model:Estimator,X_train,x_test,y_train,y_test):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model.model,X_train,y_train,cv=5,scoring=evaluation_metrics,return_train_score=True)
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
    possible_algorithms = Estimator.possible_estimators.keys()
    parser.add_argument("--model",default="all",choices=possible_algorithms)
    model_name = parser.parse_args().__dict__["model"]
    model = Estimator.possible_estimators[model_name]()
    spotify_songs = create_dataset()
    x_train,x_test,y_train,y_test = split_data(spotify_songs)   
    #save_splitted_data([train,test],["spotify_dataset_train","spotify_dataset_test"])
    #plot_valence_range(spotify_songs)
    evaluate(model,x_train,x_test,y_train,y_test)
   