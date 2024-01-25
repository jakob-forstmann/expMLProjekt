import pandas as pd 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from preprocessing import build_model
from find_DT_parameters import evaluate_experiments,perform_grid_search_cv

alpha_range = list(range(1,40,2))
alpha_range.append(0.1)
file_name = "evaluation_results/lasso_evaluation.csv"

evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__alpha":alpha_range,"model__fit_intercept":[False,True]}]
}

def get_lasso_model_for_experiments():
    return Lasso(random_state=0)

def find_lasso_parameters():
    """ evaluate a lasso model,e.g. a linear regression 
    with added regulaziation with optional intercept 
    and different regularization strength"""
    piped_model = build_model(get_lasso_model_for_experiments())
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)
   

def find_OLS_parameter():
    """ trains a ordinary Linear Regression(OLS) model
    with and without a bias.The OLS is used to train a model
    without regulazation since setting alpha to zero for the 
    lasso is discouraged in the documentation"""
    piped_liner = build_model(LinearRegression())
    piped_linear_unbiased = build_model(LinearRegression(fit_intercept=False))
    _,test_RMSE_with_bias,_,test_MEA_with_bias = evaluate_experiments(piped_liner)
    _,test_RMSE_unbiased,_,test_MEA_unbiased = evaluate_experiments(piped_linear_unbiased)
    return test_RMSE_unbiased,test_RMSE_with_bias,test_MEA_with_bias,test_MEA_unbiased
    

def plot_linear_regression_results():
    lasso_results = pd.read_csv(file_name)
    lasso_results_unbiased = lasso_results[0::2]
    print(lasso_results_unbiased)
    print("="*42)
    lasso_results_with_bias = lasso_results[1::2]
    columns_to_use = ["mean_test_neg_root_mean_squared_error","mean_test_neg_mean_absolute_error"]
    RMSE_test_unbiased  = lasso_results_unbiased[columns_to_use]
    RMSE_test_with_bias = lasso_results_with_bias[columns_to_use]
    print(f"ubiased: {RMSE_test_unbiased} \n biased {RMSE_test_with_bias}")
    test_RMSE_unbiased,test_RMSE_with_bias,test_MEA_with_bias,test_MEA_unbiased = find_OLS_parameter()
    print(test_MEA_with_bias)
    print("=")
    print(test_MEA_unbiased)
    print("RMSE")
    print(test_RMSE_unbiased)
    print(test_RMSE_with_bias)

find_lasso_parameters()
plot_linear_regression_results()