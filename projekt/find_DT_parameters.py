import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from preprocessing import build_model
from evaluation import evaluate_experiments,perform_grid_search_cv

MAX_DEPTH_UPPER = 20
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
    default_dt = build_model(get_dt_for_experiments())
    perform_grid_search_cv(evaluation_parameter,default_dt,result_file_name)

def optimze_splitter():
    """optimize the splitter type and store the results in 
    splitter_evalutation.csv"""
    piped_model = build_model(get_dt_for_experiments())
    evaluation_parameter["param_to_test"][0].pop("model__criterion")
    evaluation_parameter["param_to_test"][0]["model__splitter"] = ["random","best"]
    file_name = "evaluation_results/splitter_evaluation.csv"
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)

def optimize_min_split_samples():
    """optimize the min_samples_split hyperparameter using cross validation 
    with 5 splits. This parameter will be tested with a fixed max_depth of 7
    because tuning the parameters with Grid Search took a long time"""
    default_dt = get_dt_for_experiments()
    piped_model = build_model(default_dt)
    piped_model.set_params(model__max_depth=7)
    RMSE_results = []
    MEA_results = []
    for i in range(2,4):
        piped_model.set_params(model__min_samples_split=i)
        # TODO: decide wether to use training results 
        _,RMSE_result,_,MEA_result = evaluate_experiments(piped_model)
        RMSE_results.append(RMSE_result)
        MEA_results.append(MEA_result)
    RMSE_results = pd.Series(RMSE_results)
    MEA_results = pd.Series(MEA_results)
    print(f"RMSE mean across different min_samples_split {RMSE_results.mean()} std {RMSE_results.std()}")
    print(f" MEA  mean across different min_samples_split {MEA_results.mean()} std {RMSE_results.std()}")

def get_dt_for_experiments():
    """ always return the same decision tree that 
    can be used for testing some hyperparemeters"""
    return  DecisionTreeRegressor(random_state=0)

if __name__ =="__main__":
   optimize_min_split_samples()