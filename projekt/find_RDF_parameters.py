from sklearn.ensemble import RandomForestRegressor
from preprocessing import build_model
from evaluation import perform_grid_search_cv

MAX_DEPTH_UPPER = 30
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__n_estimators":[10,20,30]}]
}

def find_rdf_parameters():
    default_rdf = RandomForestRegressor(random_state=0)
    piped_model= build_model(default_rdf)
    file_name = "evaluation_results/rdf_parameter"
    perform_grid_search_cv(evaluation_parameter,piped_model,file_name)

if __name__ =="__main__":
    find_rdf_parameters()