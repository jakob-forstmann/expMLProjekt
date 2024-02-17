import pandas as pd 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from preprocessing import build_model
from find_DT_parameters import evaluate_experiments,perform_grid_search_cv
from dataset_statistics import plot_results


alpha_range = list(range(1,40,2))
file_name = "evaluation_results/lasso_evaluation.csv"

evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__alpha":alpha_range,"model__fit_intercept":[False,True]}]
}

def get_lasso_model_for_experiments():
    return Lasso(random_state=0)


def get_optimized_linear_model():
    return LinearRegression(fit_intercept=False)

def find_lasso_parameters():
    """ evaluate a lasso model,e.g. a linear regression 
    with added regulaziation with optional intercept 
    and different regularization strength"""
    piped_model = build_model(get_lasso_model_for_experiments())
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)
   

def train_OLS():
    """ trains a ordinary Linear Regression(OLS) model
    with and without a bias.The OLS is used to train a model
    without regulazation since setting alpha to zero for the 
    lasso is discouraged in the documentation"""
    piped_liner = build_model(LinearRegression())
    piped_linear_unbiased = build_model(LinearRegression(fit_intercept=False))
    test_RMSE_with_bias,test_MEA_with_bias = evaluate_experiments(piped_liner)
    test_RMSE_unbiased,test_MEA_unbiased = evaluate_experiments(piped_linear_unbiased)
    print("TEST RMSE WITH BIAS",round(test_RMSE_with_bias,4))
    print("test MEA  WITH BIAS",round(test_MEA_with_bias,4))
    print("TEST RMSE WITHOUT BIAS",round(test_RMSE_unbiased,4))
    print("test MEA  WITHOUT BIAS",round(test_MEA_unbiased,4))
    return test_RMSE_unbiased,test_RMSE_with_bias,test_MEA_with_bias,test_MEA_unbiased
    

def select_linear_regression_results(cv_results):
    lasso_results_unbiased = cv_results[0::2]
    lasso_results_with_bias = cv_results[1::2]
    results_description = ["Model without Intercept","Model with Interept"]
    parameter_description="alpha range"
    return [lasso_results_unbiased,lasso_results_with_bias],results_description,parameter_description
   
    
if __name__ =="__main__":
    #find_lasso_parameters()
    test_RMSE_unbiased,test_RMSE_with_bias,test_MEA_with_bias,test_MEA_unbiased = train_OLS()        
    unbiased_baselines =  {"RMSE score for OLS":test_RMSE_unbiased,"MEA score for OLS":test_MEA_unbiased}
    baselines_with_bias =  {"RMSE score for OLS":test_RMSE_with_bias,"MEA score for OLS":test_MEA_with_bias}
    plot_results(file_name,alpha_range,select_linear_regression_results,[unbiased_baselines,baselines_with_bias])
    