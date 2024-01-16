import pandas as pd 
from itertools import combinations
from models.baselines import Estimator 
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_error_scores
from preprocessing import get_column_names

def evaluate(model:Estimator,X_train,y_train):
    # criterion absolute error is deliberately left out 
    # b.c computing one cross validation already took a long time
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    param_to_test= [{"model__max_depth":list(range(1,25)),"model__criterion":["squared_error","friedman_mse"]}]
    print("param",param_to_test)
    cv_search = GridSearchCV(model,param_to_test,n_jobs=-1,scoring=evaluation_metrics,
                                refit=False,return_train_score=True)
    cv_search.fit(X_train,y_train)
    #print(cv_search.cv_results_)
    df = pd.DataFrame(cv_search.cv_results_)
    return df
   
def train(model):
    spotify_songs = create_dataset()
    columns = get_column_names()[:-1]
    columns_to_test = list(combinations(columns,len(columns)-1))
    for col_idx,columns_to_use in enumerate(columns_to_test):
        columns_to_use = list(columns_to_use)
        columns_to_use.append("valence")
        print("currently testing columns",columns_to_use)
        reduced_songs = spotify_songs[columns_to_use]
        if "track_album_name" in reduced_songs:
            vectorizer = ColumnTransformer(
                [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
            piped_model= Pipeline([("tf-idf",vectorizer),("model",model)])
        else:
            piped_model = Pipeline([("model",model)])
        print("splitting data")
        print("MODEL",model)
        x_train,_,y_train,_ = split_data(reduced_songs)
        cv_results = evaluate(piped_model,x_train,y_train)       
        cv_results.to_csv(f"evaluation_results/grid_search_cv_results_rdf{col_idx}")

def test_different_Decision_Trees():
    default_tree = DecisionTreeRegressor(random_state=0) 
    train(default_tree)
    #max_depth_RMSE_results  = evaluations_results[0:25]
  
def plot_results():
    pass
    #keys_to_use =[dataset+"_"+metric for metric in evaluation_metrics for dataset in ["mean_train","mean_test"]]
    #keys_to_use.extend(["param_criterion","param_max_depth"])
    #stripped_result = {key:cv_search.cv_results_[key]for key in keys_to_use}
    #return #pd.DataFrame(stripped_result)

if __name__ =="__main__":
    #test_different_Decision_Trees()
    test_different_RDFS()
