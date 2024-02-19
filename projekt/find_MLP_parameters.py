from sklearn.neural_network import MLPRegressor
from preprocessing import build_model
from evaluation import perform_grid_search_cv,evaluate_experiments
from dataset_statistics import plot_results


MAX_DEPTH_UPPER = 30
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{  "model__activation":["tanh","logistic"],
                                    "model__hidden_layer_sizes":[(50,100,50),(16,48,48)],
                                    "model__solver":["sgd","adam"],
                                    "model__alpha":[0.001,0.005]}]
}

def get_mlp_for_experiments():
   return  MLPRegressor(random_state=0)


def find_mlp_parameters():
    default_mlp =get_mlp_for_experiments()
    piped_model= build_model(default_mlp)
    file_name = "evaluation_results/mlp_parameter"
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)



def evaluate_default_mlp():
    default_mlp =get_mlp_for_experiments()
    piped_model= build_model(default_mlp)
    RMSE_result_30_estim,MEA_result_30_estim = evaluate_experiments(piped_model)
    return RMSE_result_30_estim,MEA_result_30_estim


if __name__ =="__main__":
    # might take a long time 
   #find_mlp_parameters()
   print(evaluate_default_mlp())