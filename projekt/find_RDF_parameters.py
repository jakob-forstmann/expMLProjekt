import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from preprocessing import build_model
from evaluation import perform_grid_search_cv,evaluate_experiments
from dataset_statistics import plot_results


MAX_DEPTH_UPPER = 30
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__n_estimators":[10,20,30]}]
}

def find_rdf_parameters():
    default_rdf =get_rdf_for_experiments()
    piped_model= build_model(default_rdf)
    file_name = "evaluation_results/rdf_parameter"
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)

def get_rdf_for_experiments():
   return  RandomForestRegressor(random_state=0)


def get_optimized_rdf():
   return  RandomForestRegressor(random_state=0,n_estimators=30)

def plot_results_rdf():
    baselines = {"RMSE":-0.1969,"MEA":-0.1511}
   
def evaluate_default_rdf():
    default_rdf = get_rdf_for_experiments()
    default_rdf_30_estim = RandomForestRegressor(random_state=0)
    piped_rdf = build_model(default_rdf)
    piped_rdf_30_estim= build_model(default_rdf_30_estim)
    RMSE_result_30_estim,MEA_result_30_estim = evaluate_experiments(piped_rdf_30_estim)
    return RMSE_result_30_estim,MEA_result_30_estim


def select_results(cv_results):
    ten_estimators = cv_results[::3]
    twenty_estimators = cv_results[1::3]
    thirty_estimators = cv_results[2::3]
    results_description = ["Ten Estimators","Twenty Estimators","30 Estimators"]
    return [ten_estimators,twenty_estimators,thirty_estimators],results_description


if __name__ =="__main__":
    # might take a long time 
    #find_rdf_parameters()
    RMSE_results_default,MEA_results_default = evaluate_default_rdf()
    baseline = {"default RDF RMSE":RMSE_results_default,"default RDF MEA":MEA_results_default}
    max_depth_range = list(range(1,MAX_DEPTH_UPPER))
    plot_results("evaluation_results/rdf_parameter.csv",max_depth_range,select_results,baseline)