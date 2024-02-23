import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from API_Keys import CLIENT_ID,CLIENT_SECRET
from models.preprocessing import get_column_names,create_dataset,sample_split_from_dataset,split_data,build_model
from experiments.find_DT_parameters import get_optimized_dt

NUMBER_OF_RECOMMENDED_SONGS = 5

def sample_songs_from_dataset():
    """ returns randomly 16 songs(=0.002%)songs from the test split 
        of the spotify dataset together with their track_id"""
    feature_columns = get_column_names()
    feature_columns.append("track_id")
    spotify_dataset = create_dataset(lambda:feature_columns)
    track_ids = spotify_dataset["track_id"]
    spotify_songs = spotify_dataset.loc[:,spotify_dataset.columns!="track_id"]
    print(spotify_songs.head(2))
    _,x_test,_,y_test = split_data(spotify_songs)
    x_test["valence"] = y_test
    x_test["track_id"] = track_ids
    return sample_split_from_dataset(0.002,x_test)

def train_DT():
    spotify_dataset = create_dataset()
    X_train,x_test,y_train,y_test = split_data(spotify_dataset)
    opt_dt = build_model(get_optimized_dt())
    opt_dt.fit(X_train,y_train)
    return opt_dt


def retrieve_recommendation(songs,valence):
    """ returns recommended songs from the spotify API.
        Note: the Spotify  API has a rate Limit"""
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET))
    songs = songs[["track_id","valence","track_name"]]
    track_recommendations = {}
    for _,song in songs.iterrows():
        track_id,_,track_name= song
        results = spotify.recommendations(seed_tracks=[track_id],target_valence=valence,limit=5)
        recommend_track_names = []
        for i in range(0,NUMBER_OF_RECOMMENDED_SONGS):
            recommend_track_names.append(results["tracks"][i]["name"])
        track_recommendations[track_name] = recommend_track_names
    return track_recommendations
    

if __name__ == "__main__":
    sampled_songs = sample_songs_from_dataset()
    gold_label_valence = sampled_songs["valence"]
    trained_DT = train_DT()
    predicted_valence = trained_DT.predict(sampled_songs)
    gold_recommended_tracks = retrieve_recommendation(sampled_songs,gold_label_valence)
    recommended_tracks_based_on_prediction = retrieve_recommendation(sampled_songs,predicted_valence)
    print("recommended tracks based on the true valence")
    print(gold_recommended_tracks)
    print("recommended tracks based on the predictedvalence")
    print(recommended_tracks_based_on_prediction)
