import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,cross_validate
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_error_scores
from preprocessing import build_model,create_dataset


MAX_DEPTH_UPPER = 30
# criterion absolute error is deliberately left out 
# b.c computing one cross validation already took a long time
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__criterion":["squared_error","friedman_mse"]}]
}
def optimize_parameter(result_file_name:str):
    """optimze a decision tree with the parameters 
    stored in evaluation_parameter using GridSearchCV"""
    default_dt = get_dt_for_experiments()
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    piped_model = build_model(default_dt)
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
    default_dt = get_dt_for_experiments()
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    piped_model = build_model(default_dt)
    piped_model.set_params(model__max_depth=7)
    RMSE_results = []
    MEA_results = []
    for i in range(2,100):
        piped_model.set_params(model__min_samples_split=i)
        results = evaluate_experiments(piped_model,X_train,y_train)
        RMSE_results.append(results["test_neg_root_mean_squared_error"].mean())
        MEA_results.append(results["test_neg_mean_absolute_error"].mean())
    RMSE_results = pd.Series(RMSE_results)
    MEA_results = pd.Series(MEA_results)
    print(f"RMSE mean across different min_samples_split {RMSE_results.mean()} std {RMSE_results.std()}")
    print(f" MEA  mean across different min_samples_split {MEA_results.mean()} std {RMSE_results.std()}")

def get_dt_for_experiments():
    return  DecisionTreeRegressor(random_state=0)

def evaluate_experiments(model,X_train,y_train):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,scoring=evaluation_metrics,cv=5)
    return pd.DataFrame(cv_score)

if __name__ =="__main__":
   optimize_min_split_samples()
    
