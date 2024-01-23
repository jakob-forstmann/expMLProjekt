import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing import create_dataset,split_data
from test_different_features import evaluate_experiments
from dataset_statistics import plot_error_scores


MAX_DEPTH_UPPER = 30
# criterion absolute error is deliberately left out 
# b.c computing one cross validation already took a long time
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__criterion":["squared_error","friedman_mse"]}]
}

def prepare_DT():
    default_tree =DecisionTreeRegressor(random_state=0)
    vectorizer = ColumnTransformer(
                [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
    return Pipeline([("tf-idf",vectorizer),("model",default_tree)])


def optimize_parameter(result_file_name:str):
    """optimze a decision tree with the parameters 
    stored in evaluation_parameter using GridSearchCV"""
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    piped_model = prepare_DT()
    scores = evaluation_parameter["evaluation_metrics"]
    param_to_test = evaluation_parameter["param_to_test"]
    cv_search = GridSearchCV(piped_model,param_to_test,n_jobs=-1,scoring=scores,
                                refit=False,return_train_score=True)
    cv_search.fit(X_train,y_train)
    df = pd.DataFrame(cv_search.cv_results_)
    df.to_csv(result_file_name)

def optimze_splitter():
    """optimize the splitter type and store the results in 
    splitter_evalutation.csv"""
    evaluation_parameter["param_to_test"][0].pop("model__criterion")
    evaluation_parameter["param_to_test"][0]["model__splitter"] = ["random","best"]
    optimize_parameter("evaluation_results/splitter_evaluation.csv")

def optimize_min_split_samples():
    """optimize the min_samples_split hyperparameter using cross validation 
    with 5 splits. This parameter will be tested with a fixed max_depth of 7
    because tuning the parameters with Grid Search took a long time"""
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    piped_model = prepare_DT()
    piped_model.set_params(decision_tree__max_depth=7)
    for i in range(2,20):
        piped_model.set_params(decision_tree__min_samples_split=i)
        results = evaluate_experiments(X_train,y_train)
        results.to_csv(f"evaluation_parameter/min_samples{i}")

                    
if __name__ =="__main__":
   optimze_splitter()
    
