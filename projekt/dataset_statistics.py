from preprocessing import create_dataset
import pandas as pd 
import matplotlib.pyplot as plt

def plot_valence_range():
    spotify_songs = create_dataset()
    grouped_values = pd.cut(spotify_songs["valence"],bins=5)
    grouped_data = spotify_songs.groupby(grouped_values,observed=False).count()
    grouped_data = grouped_data["valence"]
    plot_bar(grouped_data,"valence range","number of datapoints")

def plot_bar(data,x_label,y_label):
    ax = data.plot.bar(figsize=(18,18),rot=1)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.show()

def plot_valence_frequencies():
    distribution = count_unique_valence_values()
    grouped_frequencies = distribution.value_counts().sort_values(ascending=False)
    print(grouped_frequencies.head(20))
    first_twenty_most_freq_vals = distribution.iloc[0:20]
    grouped_frequencies2 = grouped_frequencies[3:]
    plot_bar(grouped_frequencies2,"number of occurence","frequency")

def plot_error_scores(error_scores,parameter,plot_title,x_label,y_label):
    plt.figure(figsize=(12, 6))
    plt.plot(parameter, error_scores, marker='o', linestyle='-',linewidth=3,color='b')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks(parameter)
    #plt.legend()
    plt.ylim(0,1.2)
    plt.grid(True)
    plt.show()



def calculate_most_frequent_value():
    unique_combinations = count_unique_valence_values()
    return unique_combinations.index[0]

def count_unique_valence_values():
    spotify_songs = create_dataset()
    unique_combinations = spotify_songs.groupby("valence").size().sort_values(ascending=False)
    unique_combinations.to_csv("combinations_of_values.csv")
    return unique_combinations

if __name__=="__main__":
    #plot_valence_range()
    plot_valence_frequencies()