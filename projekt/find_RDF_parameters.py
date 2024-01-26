import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from preprocessing import build_model
from evaluation import perform_grid_search_cv
from dataset_statistics import plot_error_scores


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


def plot_results_rdf():
    scores = evaluation_parameter["evaluation_metrics"]
    columns_to_use = [dataset+"_"+metric for dataset in ["mean_train","mean_test"] for metric in scores]   
    cv_results = pd.read_csv("evaluation_results/rdf_parameter_max_depth_30.csv")  
    columns_to_use.append("param_model__max_depth")
    MSE_results = cv_results[columns_to_use]
    ten_estimators = MSE_results[::3]
    twenty_estimators = MSE_results[1::3]
    thirty_estimators = MSE_results[2::3]
    train_results_ten =  ten_estimators[columns_to_use[0:2]]
    test_results_ten  = ten_estimators[columns_to_use[2:4]]
    train_results_twenty=  twenty_estimators[columns_to_use[0:2]]
    test_results_twenty  = twenty_estimators[columns_to_use[2:4]]
    train_results_thirty=  thirty_estimators[columns_to_use[0:2]]
    test_results_thirty  = thirty_estimators[columns_to_use[2:4]]
    plot_error_scores(train_results_ten,MSE_results["param_model__max_depth"][::3],
    "Max_depth on the training dataset with 10 estimators ","max_depth","mean across 5 folds")
    plot_error_scores(test_results_ten,MSE_results["param_model__max_depth"][::3],
    "Max_depth on the evaluation dataset with 10 estimators ","max_depth","mean MSE across 5 folds")
    plot_error_scores(train_results_twenty,MSE_results["param_model__max_depth"][1::3],
    "Max_depth on the training dataset with 20 estimators","max_depth","mean MSE across 5 folds")
    plot_error_scores(test_results_twenty,MSE_results["param_model__max_depth"][1::3],
    "Max_depth on the evaluation dataset with 20 estimators ","max_depth","mean MSE across 5 folds")
    plot_error_scores(train_results_thirty,MSE_results["param_model__max_depth"][2::3],
    "Max_depth on the training dataset with 30 estimators","max_depth","mean MSE across 5 folds")
    plot_error_scores(test_results_thirty,MSE_results["param_model__max_depth"][2::3],
    "Max_depth on the evaluation dataset with 30 estimators","max_depth","mean MSE across 5 folds")



if __name__ =="__main__":
    #find_rdf_parameters()
    plot_results_rdf()