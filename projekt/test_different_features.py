from itertools import combinations
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from preprocessing import create_dataset,get_column_names,split_data

def train_with_different_feature_combinations(model):
    spotify_songs = create_dataset()
    columns = get_column_names()
    columns_to_test = list(combinations(columns[:-1],len(columns)-1))
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
        cv_results.to_csv(f"evaluation_results/grid_search_cv_results_new{col_idx}")


def evaluate(model,X_train,y_train):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,scoring=evaluation_metrics,cv=5)
    print(f"Evaulating with model {str(model)}")
    print("Training Dataset")
    print(f"Model{model}: root mean squarred error",cv_score["test_neg_root_mean_squared_error"])
    print("mean across 5 folds",cv_score["test_neg_root_mean_squared_error"].mean())
    print(f"Model{model}: mean absolute error",cv_score["test_neg_mean_absolute_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())
    return DataFrame(cv_score)