from sklearn.neural_network import MLPRegressor
from models.preprocessing import build_model
from models.evaluation import perform_grid_search_cv,evaluate_experiments

evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{  "model__activation":["logistic"],
                                    "model__hidden_layer_sizes":[(100,100),(100,100,100),(100,100,100,100)],
                                    "model__solver":["adam"],
                                    "model__alpha":[0.01,0.1]}]
}


def get_mlp_for_experiments():
    """ a helper function that always return the same MLP.
        Used for testing some hyperparemeters"""
    return  MLPRegressor(random_state=0)

def get_optimized_mlp():
    """ helper function to get the MLP with 
        the best found hyperparameters"""
    return MLPRegressor(hidden_layer_sizes=(100,100),activation="logistic",alpha=0.1)

def find_mlp_parameters():
    """performs grid search CV with the parameters above 
        to evaluate find the best set of hyperparameters 
        from these parameters """
    default_mlp = get_mlp_for_experiments()
    piped_model= build_model(default_mlp)
    file_name = "../evaluation_results/mlp_parameter44"
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)

def evaluate_default_mlp():
    """trains a MLP with the default 
        arguments provided by sklearn on the dataset."""
    default_mlp =get_mlp_for_experiments()
    piped_model= build_model(default_mlp)
    RMSE_result_30_estim,MEA_result_30_estim = evaluate_experiments(piped_model)
    return RMSE_result_30_estim,MEA_result_30_estim


if __name__ =="__main__":
    # might take a long time 
   find_mlp_parameters()
   print(evaluate_default_mlp())
