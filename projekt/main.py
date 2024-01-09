from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
import argparse
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_valence_range
from models.baselines import Estimator 

def evaluate(model:Estimator,X_train,x_test,y_train,y_test):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,scoring=evaluation_metrics,cv=5)
    print(f"Evaulating with model {str(model)}")
    print("Training Dataset")
    print(f"Model{model}: root mean squarred error",cv_score["test_neg_root_mean_squared_error"])
    print("mean across 5 folds",cv_score["test_neg_root_mean_squared_error"].mean())
    print(f"Model{model}: mean absolute error",cv_score["test_neg_mean_absolute_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())
    print("Test Dataset")
    model.fit(X_train,y_train)
    prediction = model.predict(x_test)
    print(f"Model{model}: root mean squarred error",mean_squared_error(y_test,prediction,squared=False))
    print(f"Model{model}: mean absolute error",mean_absolute_error(y_test,prediction))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="evaluate with different algorithms")
    possible_algorithms = Estimator.get_estimators()
    parser.add_argument("--model",choices=possible_algorithms)
    model_name = parser.parse_args().__dict__["model"]
    model = Estimator.possible_estimators[model_name]().get_model()
    spotify_songs = create_dataset()
    x_train,x_test,y_train,y_test = split_data(spotify_songs)   
    evaluate(model,x_train,x_test,y_train,y_test)
    