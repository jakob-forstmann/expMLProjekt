from itertools import combinations
import pandas as pd 
from sklearn.pipeline import Pipeline
from models.preprocessing import build_model,get_column_names
from models.evaluation import evaluate_experiments
from experiments.find_DT_parameters import get_optimized_dt
from experiments.find_lasso_parameters import get_optimized_linear_model
from experiments.find_RDF_parameters import get_optimized_rdf


def train_with_different_feature_combinations(model,select_columns,model_name):
    """ reports the RMSE and the MEA for different feature combinations"""
    columns_to_test = select_columns()
    RMSE = []
    MEA  = []
    combinations_results = pd.DataFrame({"columns":columns_to_test})
    for columns_to_use in columns_to_test:
        columns_to_use = list(columns_to_use)
        columns_to_use.append("valence")
        if "track_name" in columns_to_use:
           piped_model = build_model(model)
        else:
            piped_model = Pipeline([("model",model)])
        test_RMSE,test_MEA = evaluate_experiments(piped_model,columns_to_use)
        RMSE.append(round(test_RMSE,4))
        MEA.append(round(test_MEA,4))
    combinations_results["RMSE across 5 folds"] = RMSE
    combinations_results["MAE across 5 folds"]  = MEA 
    combinations_results.to_csv(f"../evaluation_results/feature_combinations_{len(columns_to_test)}_{model_name}.csv")

def test_combinations_of_four_columns():
    """ returns all combinations of four columns from
    the five used columns"""
    columns = get_column_names()
    return list(combinations(columns[:-1],len(columns)-2))
   
def test_combinations_of_three_columns():
    columns = get_column_names()
    return list(combinations(columns[:-1],len(columns)-3))

if __name__=="__main__":
    opt_dt = get_optimized_dt()
    opt_linear_model = get_optimized_linear_model()
    opt_rdf = get_optimized_rdf()
    train_with_different_feature_combinations(opt_linear_model,test_combinations_of_three_columns,"linear")
    train_with_different_feature_combinations(opt_linear_model,test_combinations_of_four_columns,"linear")
    train_with_different_feature_combinations(opt_dt,test_combinations_of_three_columns,"dt")
    train_with_different_feature_combinations(opt_dt,test_combinations_of_four_columns,"dt")
    train_with_different_feature_combinations(opt_rdf,test_combinations_of_three_columns,"rdf")
    train_with_different_feature_combinations(opt_rdf,test_combinations_of_four_columns,"rdf")