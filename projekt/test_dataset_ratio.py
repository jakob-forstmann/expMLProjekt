import numpy as np 
import pandas as pd
from preprocessing import sample_split_from_dataset,build_model
from find_DT_parameters import get_optimized_dt
from find_RDF_parameters import get_optimized_rdf
from find_lasso_parameters import get_optimized_linear_model
from evaluation import evaluate_experiments
from dataset_statistics import plot_error_scores


def test_different_dataset_sizes(model):
    piped_model = build_model(model)
    MEA_results = []
    RMSE_results = []
    for size in list(range(10,110,10)):
        RMSE_result,MEA_result = evaluate_experiments(piped_model,dataset_loader=lambda:sample_split_from_dataset(size/100))
        RMSE_results.append(round(RMSE_result,4))
        MEA_results.append(round(MEA_result,4))
    return RMSE_results,MEA_results


if __name__ =="__main__":
    opt_dt = get_optimized_dt()
    opt_linear = get_optimized_linear_model()
    opt_rdf = get_optimized_rdf()
    dt_RMSE,dt_MAE = test_different_dataset_sizes(opt_dt)
    rdf_RMSE,rdf_MAE = test_different_dataset_sizes(opt_rdf)
    linear_RMSE,linear_MAE = test_different_dataset_sizes(opt_linear)
    scores = pd.DataFrame({ "DT RMSE":dt_RMSE,"DT MAE":dt_MAE,
                            "RDF RMSE":dt_RMSE,"RDF MAE":rdf_MAE,
                            "linear RMSE":linear_RMSE,"linear MAE":linear_MAE})
    plot_error_scores(scores,list(range(10,110,10)),"Dataset size percentage","percentage of the dataset")
    