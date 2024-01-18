import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing import create_dataset,split_data


MAX_DEPTH_UPPER = 20
evaluation_parameter = {
                "evaluation_metrics": ["neg_root_mean_squared_error","neg_mean_absolute_error"],
                "param_to_test":[{"model__max_depth":list(range(1,MAX_DEPTH_UPPER)),
                "model__n_estimators":[10,20,30]}]
}

def find_rdf_parameters():
    default_tree = RandomForestRegressor(random_state=0)
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    vectorizer = ColumnTransformer(
                [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
    piped_model= Pipeline([("tf-idf",vectorizer),("model",default_tree)])
   
    scores = evaluation_parameter["evaluation_metrics"]
    param_to_test = evaluation_parameter["param_to_test"]
    cv_search = GridSearchCV(piped_model,param_to_test,n_jobs=-1,scoring=scores,
                                refit=False,return_train_score=True)
    cv_search.fit(X_train,y_train)
    df = pd.DataFrame(cv_search.cv_results_)
    df.to_csv("evaluation_results/rdf_parameter")
    return df

if __name__ =="__main__":
    find_rdf_parameters()