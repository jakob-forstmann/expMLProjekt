from itertools import combinations
import pandas as pd 
from sklearn.pipeline import Pipeline
from preprocessing import create_dataset,get_column_names,split_data,build_model
from find_DT_parameters import get_dt_for_experiments,evaluate_experiments


def train_with_different_feature_combinations(model,select_columns):
    spotify_songs = create_dataset()
    columns_to_test = select_columns()
    for col_idx,columns_to_use in enumerate(columns_to_test):
        columns_to_use = list(columns_to_use)
        columns_to_use.append("valence")
        print("currently testing columns",columns_to_use)
        reduced_songs = spotify_songs[columns_to_use]
        if "track_album_name" in reduced_songs:
           piped_model = build_model(model)
        else:
            piped_model = Pipeline([("model",model)])
        print("splitting data")
        print("MODEL",model)
        x_train,_,y_train,_ = split_data(reduced_songs)
        cv_results = evaluate_experiments(piped_model,x_train,y_train)    
        cv_results.to_csv(f"evaluation_results/grid_search_cv_results_{len(columns_to_test)}_combinations_{col_idx}")


def test_combinations_of_four_columns():
    columns = get_column_names()
    return list(combinations(columns[:-1],len(columns)-1))

def test_combinations_of_three_columns():
    columns = [("track_album_name","mode","tempo"),"track_album_name",""]
    columns = get_column_names()
    return columns

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
        print(f"column names:{column_name} feature importance: {columns_importance[col_idx]}") 
    tf_idf_values = pd.Series(feat_importance[0:len(feat_importance)-4]).sort_values(ascending=False)[0:10]
    top_words_tf_idf = [feature_names[idx] for idx in tf_idf_values.index]
    tf_idf_data = {"word":top_words_tf_idf,"TF-IDF":tf_idf_values}
    top_words_tf_idf_scores = pd.DataFrame(tf_idf_data)
    print(top_words_tf_idf_scores)
   

if __name__=="__main__":
    default_dt = get_dt_for_experiments()
    calculate_feature_importance(default_dt)
    #train_with_different_feature_combinations(default_dt,test_combinations_of_three_columns)