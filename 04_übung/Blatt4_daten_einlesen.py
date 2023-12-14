import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 

def read_entire_dataset(file_name:str)-> pd.DataFrame:
    # the columns accoustiness,liveness and valence contain european 
    # decimals(,) but only the column valence is revelant here
    converter = {"valence":lambda x:np.float64(x.replace(",","."))}
    return pd.read_csv(file_name,converters=converter)

def keep_necessary_columns(songs:pd.DataFrame,columns_to_keep:[str]):
    return songs[columns_to_keep]

def split_data(songs:pd.DataFrame):
    training_data,test_data = train_test_split(songs,test_size=0.25,random_state=42)
    return (training_data,test_data)

def save_splitted_data(splitted_datasets:[np.array],file_names:[str]):
    for dataset_name,file_name in zip(splitted_datasets,file_names):
        dataset = pd.DataFrame(dataset_name)
        dataset.to_csv(file_name)

def rescale_data_range(songs:pd.DataFrame,columns_to_scale:[np.array]):
    """ converts a value between 0-100 to a double between 0 and 1 
    for the passed columns"""
    cleaned = songs["danceability"].apply(lambda val:pd.to_numeric(val,errors="coerce"))
    songs = songs.drop(columns=["danceability"])
    songs.insert(0,"danceability",cleaned)
    songs.dropna(inplace=True)
    songs.where(songs[columns_to_scale]<1,songs[columns_to_scale]/1000,inplace=True)
    return songs

def plot_valence_range(spotify_songs:pd.DataFrame):
    grouped_values,bins= pd.cut(spotify_songs["valence"],bins=5,retbins=True)
    grouped_data = spotify_songs.groupby(grouped_values,observed=False).count()
    grouped_data = grouped_data["valence"]
    ax = grouped_data.plot.bar(figsize=(8,8),legend=False,rot=1)
    ax.set_ylabel("number of datapoints")
    ax.set_xlabel("valence range")
    plt.show()


def preprocess_dataset():
    columns_to_use = ["danceability","tempo","loudness","mode","track_album_name","valence"]
    spotify_songs_complete  = read_entire_dataset("../data/spotify_songs.csv")
    spotify_songs  = keep_necessary_columns(spotify_songs_complete,columns_to_use)
    spotify_songs  = rescale_data_range(spotify_songs,["danceability","valence"])
    return spotify_songs

if __name__=="__main__":
    spotify_songs = preprocess_dataset()
    train,test = split_data(spotify_songs)
    save_splitted_data([train,test],["spotify_dataset_train","spotify_dataset_test"])
    plot_valence_range(spotify_songs)
