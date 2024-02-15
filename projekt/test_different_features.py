from itertools import combinations
import pandas as pd 
from sklearn.pipeline import Pipeline
from preprocessing import create_dataset,get_column_names,split_data,build_model
from find_DT_parameters import get_optimized_dt
from find_lasso_parameters import get_optimized_linear_model
from find_RDF_parameters import get_optimized_rdf
from evaluation import evaluate_experiments


def train_with_different_feature_combinations(model,select_columns):
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
        _,test_RMSE,_,test_MEA = evaluate_experiments(piped_model,columns_to_use)
        RMSE.append(round(test_RMSE,4))
        MEA.append(round(test_MEA,4))
    combinations_results["RMSE across 5 folds"] = RMSE
    combinations_results["MAE across 5 folds"]  = MEA 
    combinations_results.to_csv(f"evaluation_results/feature_combinations_{len(columns_to_test)}_dt2")

def test_combinations_of_four_columns():
    """ returns all combinations of four columns from
    the five used columns"""
    columns = get_column_names()
    return list(combinations(columns[:-1],len(columns)-2))
   
def test_combinations_of_three_columns():
    columns = get_column_names()
    return list(combinations(columns[:-1],len(columns)-3))

def calculate_feature_importance(model):
    spotify_songs = create_dataset()
    piped_model = build_model(model)
    x_train,_,y_train,_ = split_data(spotify_songs)
    piped_model.fit(x_train,y_train)
    fitted_model = piped_model[1]
    vect = piped_model[0]
    feature_names = vect.get_feature_names_out()
    column_names  = feature_names[-4:]
    feat_importance= fitted_model.feature_importances_
    columns_importance = feat_importance[-4:]
    for col_idx,column_name in enumerate(column_names):
        print(f"column names:{column_name} feature importance: {round(columns_importance[col_idx],4)}") 
    tf_idf_values = pd.Series(feat_importance[0:len(feat_importance)-4]).sort_values(ascending=False)[0:10]
    top_words_tf_idf = [feature_names[idx] for idx in tf_idf_values.index]
    tf_idf_data = {"word":top_words_tf_idf,"TF-IDF":tf_idf_values}
    top_words_tf_idf_scores = pd.DataFrame(tf_idf_data)
    print(top_words_tf_idf_scores)
   

if __name__=="__main__":
    opt_dt = get_optimized_dt()
    opt_linear_model = get_optimized_linear_model()
    opt_rdf = get_optimized_rdf()
    calculate_feature_importance(opt_dt)
    print(opt_rdf)
    print("RDF Feature Importance")
    #calculate_feature_importance(opt_rdf)
    #train_with_different_feature_combinations(opt_dt,test_combinations_of_three_columns)
    #train_with_different_feature_combinations(opt_dt,test_combinations_of_four_columns)

