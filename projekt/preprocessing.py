from functools import cache
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 

def read_entire_dataset(file_name:str)-> pd.DataFrame:
    # the columns accoustiness,liveness and valence contain european 
    # decimals(,) but only the column valence is revelant here
    converter = {"valence":lambda x:np.float64(x.replace(",","."))}
    return pd.read_csv(file_name,converters=converter)

def keep_necessary_columns(songs:pd.DataFrame,columns_to_keep:[str]):
    return songs[columns_to_keep]

def split_data(songs:pd.DataFrame):
    album_names = songs["track_album_name"].values
    songs["track_album_name"] = vectorize_album_name(album_names)
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

def vectorize_album_name(album_names:list[str]):
    """represents each track_album name using tf-idf
    returns a list with the representation for each track album name
    """
    vectorizer = TfidfVectorizer()
    # each song is a document, the track_album_name is the term
    document_term_matrix =  vectorizer.fit_transform(album_names)
    return list(document_term_matrix.toarray())

@cache
def create_dataset():
    columns_to_use = ["danceability","track_album_name","tempo","loudness","mode","valence"]
    spotify_songs_complete  = read_entire_dataset("data/spotify_songs.csv")
    spotify_songs  = keep_necessary_columns(spotify_songs_complete,columns_to_use)
    spotify_songs  = rescale_data_range(spotify_songs,["danceability","valence"])
    return spotify_songs

