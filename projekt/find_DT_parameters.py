import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from preprocessing import build_model
from evaluation import evaluate_experiments,perform_grid_search_cv
from dataset_statistics import plot_error_scores

MAX_DEPTH_UPPER = 20
# criterion absolute error is deliberately left out 
# b.c computing one cross validation already took a long time
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__criterion":["squared_error","friedman_mse"]}]
}
def optimize_parameter():
    """optimze a decision tree with the parameters 
    stored in evaluation_parameter using GridSearchCV"""
    result_file_name = "evaluation_results/dt_evaluation2.csv"
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

def plot_results_dt():
    cv_results = pd.read_csv("evaluation_results/dt_parameter.csv")
    scores = evaluation_parameter["evaluation_metrics"]
    columns_to_use = [dataset+"_"+metric for dataset in ["mean_train","mean_test"] for metric in scores]
    print(columns_to_use)
    columns_to_use.append("param_model__max_depth")
    cv_results = cv_results[columns_to_use]
    MSE_results = cv_results[0:MAX_DEPTH_UPPER-1]
    FMSE_results = cv_results[MAX_DEPTH_UPPER-1:]
    print(MSE_results)
    print("="*42)
    print(FMSE_results)
    MSE_train_results = MSE_results[columns_to_use[0:2]]
    MSE_test_results  = MSE_results[columns_to_use[2:4]]
    FMSE_train_results = FMSE_results[columns_to_use[0:2]]
    FMSE_test_results  = FMSE_results[columns_to_use[2:4]]
    plot_error_scores(MSE_train_results,MSE_results["param_model__max_depth"],"Max_depth on the training dataset ","max_depth","mean across 5 folds")
    plot_error_scores(MSE_test_results,MSE_results["param_model__max_depth"],"Max_depth on the evaluation dataset ","max_depth","mean across 5 folds")
    plot_error_scores(FMSE_train_results,MSE_results["param_model__max_depth"],"Max_depth on the training dataset ","max_depth","Mean  across 5 Folds")
    plot_error_scores(FMSE_test_results,MSE_results["param_model__max_depth"],"Max_depth on the evaluation dataset ","max_depth","Mean  across 5 Folds")


if __name__ =="__main__":
    optimize_parameter()
    optimze_splitter()
    optimize_min_split_samples()