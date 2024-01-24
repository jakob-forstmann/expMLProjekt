import pandas as pd 
import numpy as np 
from functools import cache
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def read_entire_dataset(file_name:str)-> pd.DataFrame:
    # the columns accoustiness,liveness and valence contain european 
    # decimals(,) but only the column valence is revelant here
    converter = {"valence":lambda x:np.float64(x.replace(",","."))}
    return pd.read_csv(file_name,converters=converter)

def keep_necessary_columns(songs:pd.DataFrame,columns_to_keep:[str]):
    return songs[columns_to_keep]

def split_data(songs:pd.DataFrame):
    X = songs.loc[:,songs.columns!="valence"]
    labels = songs["valence"]
    return train_test_split(X,labels,test_size=0.25,random_state=42)

def save_splitted_data(splitted_datasets:[np.array],file_names:[str]):
    for dataset_name,file_name in zip(splitted_datasets,file_names):
        dataset = pd.DataFrame(dataset_name)
        dataset.to_csv(file_name)

def remove_strings_from_numeric_columns(songs:pd.DataFrame):
    columns_to_clean = songs.loc[:,songs.columns!="track_album_name"]
    cleaned = columns_to_clean.apply(lambda val:pd.to_numeric(val,errors="coerce")).dropna()
    cleaned.insert(0,"track_album_name",songs["track_album_name"])
    return cleaned

def rescale_data_range(songs:pd.DataFrame,columns_to_scale:[np.array]):
    """ converts a value between 0-100 to a double between 0 and 1 
    for the passed columns"""
    songs = songs.dropna()
    songs = remove_strings_from_numeric_columns(songs)
    songs.where(songs[columns_to_scale]<1,songs[columns_to_scale]/1000,inplace=True)
    return songs

def build_model(model):
    vectorizer = ColumnTransformer(
                [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
    return Pipeline([("tf-idf",vectorizer),("model",model)])

@cache
def create_dataset():
    columns_to_use = get_column_names()
    spotify_songs_complete  = read_entire_dataset("data/spotify_songs.csv")
    spotify_songs  = keep_necessary_columns(spotify_songs_complete,columns_to_use)
    spotify_songs  = rescale_data_range(spotify_songs,["danceability","valence"])
    return spotify_songs

def get_column_names():
    return ["danceability","track_album_name","tempo","loudness","mode","valence"]