import pandas as pd 
import numpy as np 
from functools import cache
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def read_entire_dataset(file_name:str)-> pd.DataFrame:
    """reads the dataset from a csv file.Returns the dataset as a Dataframe"""
    # the columns accoustiness,liveness and valence contain european 
    # decimals(,) but only the column valence is revelant here
    converter = {"valence":lambda x:np.float64(x.replace(",","."))}
    return pd.read_csv(file_name,converters=converter)

def keep_necessary_columns(songs:pd.DataFrame,columns_to_keep:[str]):
    """delete all columns from the songs parameters except 
    for the columns_to_keep"""
    return songs[columns_to_keep]

def split_data(songs:pd.DataFrame):
    """split data in 75% and 25% of the spotify dataset"""
    X = songs.loc[:,songs.columns!="valence"]
    labels = songs["valence"]
    return train_test_split(X,labels,test_size=0.25,random_state=42)

def save_splitted_data(splitted_datasets:[np.array],file_names:[str]):
    for dataset_name,file_name in zip(splitted_datasets,file_names):
        dataset = pd.DataFrame(dataset_name)
        dataset.to_csv(file_name)

def remove_strings_from_numeric_columns(songs:pd.DataFrame):
    """remove non numeric values from the numeric columns"""
    columns_to_clean = songs.loc[:,songs.columns!="track_name"]
    cleaned = columns_to_clean.apply(lambda val:pd.to_numeric(val,errors="coerce")).dropna()
    cleaned.insert(0,"track_name",songs["track_name"])
    return cleaned

def rescale_data_range(songs:pd.DataFrame,columns_to_scale:[np.array],threshold=1,scal_factor=1000):
    """ converts a value between 0-100 to a double between 0 and 1 
    for the passed columns"""
    songs = songs.dropna()
    songs = remove_strings_from_numeric_columns(songs)
    songs.where(songs[columns_to_scale]<threshold,songs[columns_to_scale]/scal_factor,inplace=True)
    return songs

def build_model(model):
    """build a model Pipeline with a TF-IDF Vectorizer for the 
    track names and the passed model as the second step"""
    vectorizer = ColumnTransformer(
                [("TF-IDF",TfidfVectorizer(),"track_name")],remainder="passthrough")
    return Pipeline([("tf-idf",vectorizer),("model",model)])

@cache
def create_dataset():
    """ builds the cleaned dataset.Removes unnecesary columns and 
    junk data from the columns"""
    columns_to_use = get_column_names()
    spotify_songs_complete  = read_entire_dataset("data/spotify_songs.csv")
    spotify_songs  = keep_necessary_columns(spotify_songs_complete,columns_to_use)
    spotify_songs  = rescale_data_range(spotify_songs,["danceability","valence"],)
    spotify_songs  = rescale_data_range(spotify_songs,["tempo"],threshold=230)
    return spotify_songs

def get_column_names():
    """a helper function to get all used columns from the spotify dataset"""
    return ["danceability","track_name","tempo","key","mode","valence"]


def sample_split_from_dataset(percentage=0.1):
    """ sample the passed percentrage from the dataset.
    Useful to test which percentage of the dataset is sufficient"""
    spotify_songs = create_dataset()
    return spotify_songs.sample(frac=percentage,random_state=0)

def save_cleaned_dataset():
    spotify_songs = create_dataset()
    spotify_songs.to_csv("data/cleaned_data.csv")


