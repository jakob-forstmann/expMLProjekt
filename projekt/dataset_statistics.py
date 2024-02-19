from preprocessing import create_dataset
import pandas as pd 
import matplotlib.pyplot as plt

def plot_valence_range():
    spotify_songs = create_dataset()
    grouped_values = pd.cut(spotify_songs["valence"],bins=5)
    grouped_data = spotify_songs.groupby(grouped_values,observed=False).count()
    grouped_data = grouped_data["valence"]
    create_bar_plot(grouped_data,"valence Intervall","Anzahl an Datenpunkten")

def plot_valence_frequencies():
    distribution = count_unique_valence_values()
    grouped_frequencies = distribution.value_counts().sort_values(ascending=False)
    grouped_frequencies = grouped_frequencies[3:]
    create_bar_plot(grouped_frequencies,"number of occurence","frequency")


def plot_error_scores(error_scores,parameter,plot_title,x_label,stds=None,baselines=None):
    plt.figure(figsize=(12, 6))
    if baselines is not None:
        for baseline_label,baseline in baselines.items():
            baselines_values = [baseline for _ in range(0,len(parameter))]
            plt.plot(parameter,baselines_values,marker='o', linestyle='-',linewidth=3,label=baseline_label)   
    labels = error_scores.columns
    for column_idx in range(0,len(error_scores.columns)):
        plt.plot(parameter,error_scores.iloc[:,column_idx],marker='o', linestyle='-',linewidth=3,label=labels[column_idx])
        if stds is not None:
            std_bottom = error_scores.iloc[:,column_idx]-stds.iloc[:,column_idx]
            std_upper  = error_scores.iloc[:,column_idx]+stds.iloc[:,column_idx]
            plt.fill_between(parameter,std_bottom,std_upper,alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel("mean across 5 folds")
    plt.title(plot_title)
    plt.xticks(parameter)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_results(file_name,parameter_range,select_results,baselines=[None,None]):
    cv_results = pd.read_csv(file_name)
    columns_to_use = [ "mean_test_neg_root_mean_squared_error","mean_test_neg_mean_absolute_error",
                       "std_test_neg_root_mean_squared_error","std_test_neg_mean_absolute_error"] 
    cv_results = cv_results[columns_to_use] 
    results,results_description,parameter_description = select_results(cv_results)
    for results,criterion,baseline in zip(results,results_description,baselines): 
        test_results = results[columns_to_use[0:2]]
        std =  results[columns_to_use[2:4]]
        test_results.columns = ["RMSE","MAE"]
        plot_error_scores(test_results,parameter_range,f"{criterion} on the evaluation dataset",
                          parameter_description,stds = std,baselines=baseline)


def create_bar_plot(data,x_label,y_label):
    ax = data.plot.bar(figsize=(18,18),rot=1)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.show()


def calculate_most_frequent_value():
    unique_combinations = count_unique_valence_values()
    return unique_combinations.index[0]

def count_unique_valence_values():
    spotify_songs = create_dataset()
    unique_combinations = spotify_songs.groupby("valence").size().sort_values(ascending=False)
    return unique_combinations



