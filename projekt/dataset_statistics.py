import pandas as pd 
import matplotlib.pyplot as plt

def plot_valence_range(spotify_songs:pd.DataFrame):
    grouped_values,bins= pd.cut(spotify_songs["valence"],bins=5,retbins=True)
    grouped_data = spotify_songs.groupby(grouped_values,observed=False).count()
    grouped_data = grouped_data["valence"]
    ax = grouped_data.plot.bar(figsize=(8,8),legend=False,rot=1)
    ax.set_ylabel("number of datapoints")
    ax.set_xlabel("valence range")
    plt.show()

def calculate_most_frequent_value(spotify_songs:pd.DataFrame):
    unique_combinations = count_unique_valence_values(spotify_songs)
    return unique_combinations.index[0]

def count_unique_valence_values(spotify_songs):
    unique_combinations = spotify_songs.groupby("valence").size().sort_values(ascending=False)
    unique_combinations.to_csv("combinations_of_values.csv")
    return unique_combinations
