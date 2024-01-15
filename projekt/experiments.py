import pandas as pd 
from models.baselines import Estimator 
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_error_scores

def evaluate(model:Estimator,X_train,y_train):
    # criterion absolute error is deliberately left out 
    # b.c computing one cross validation already took a long time
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    param_to_test= {"max_depth":list(range(1,25)),"criterion":["squared_error","friedman_mse"]}
    cv_search = GridSearchCV(model,param_to_test,n_jobs=-1,scoring=evaluation_metrics,
                                refit=False,return_train_score=True)
    cv_search.fit(X_train,y_train)
    #print(cv_search.cv_results_)
    keys_to_use =[dataset+"_"+metric for metric in evaluation_metrics for dataset in ["mean_train","mean_test"]]
    keys_to_use.extend(["param_criterion","param_max_depth"])
    stripped_result = {key:cv_search.cv_results_[key]for key in keys_to_use}
    return pd.DataFrame(stripped_result)
   
def train(model):
    spotify_songs = create_dataset()
    spotify_songs.drop(columns="track_album_name",inplace=True)
    if "track_album_name" in spotify_songs:
        vectorizer = ColumnTransformer(
            [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
        model = make_pipeline(vectorizer,model)
    print("splitting data")
    x_train,_,y_train,_ = split_data(spotify_songs)
    return evaluate(model,x_train,y_train)       

def test_different_Decision_Trees():
    default_tree = DecisionTreeRegressor(random_state=0) 
    evaluations_results = train(default_tree)
    max_depth_RMSE_results  = evaluations_results[0:25]
  
if __name__ =="__main__":
    test_different_Decision_Trees()
