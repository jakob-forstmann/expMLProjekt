from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate
from preprocessing import create_dataset,split_data
from dataset_statistics import plot_valence_range,calculate_most_frequent_value

def load_dataset():
    spotify_songs = create_dataset()
    return split_data(spotify_songs)

def baselines(most_frequent:float):
    # always predict mean instead of most common class
    #  b.c. we deal with a regression problem
    mean_classifier = DummyRegressor(strategy="mean")
    #  instead of using the most frequent class use the most 
    # frequent value for valence
    most_frequent_classifier = DummyRegressor(strategy="constant",constant=most_frequent)
    return(mean_classifier,most_frequent_classifier)

def evaluate(model,X_train,y_train,x_test,y_test):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,cv=5,scoring=evaluation_metrics,return_train_score=True)
    print("Training Dataset:")
    print(f"Model{model}: root mean squarred error",cv_score["train_neg_root_mean_squared_error"])
    print(f"Model{model}: mean absolute error",cv_score["train_neg_mean_absolute_error"])
    print("Test Dataset:")
    print(f"Model{model}: root mean squarred error",cv_score["test_neg_root_mean_squared_error"])
    print(f"Model{model}: mean absolute error",cv_score["test_neg_mean_absolute_error"])

if __name__=="__main__":
    spotify_songs = create_dataset()
    x_train,x_test,y_train,y_test = split_data(spotify_songs)   
    #save_splitted_data([train,test],["spotify_dataset_train","spotify_dataset_test"])
    #plot_valence_range(spotify_songs)
    frequent_value = calculate_most_frequent_value(spotify_songs)
    for baseline_algorithm in baselines(frequent_value):
        evaluate(baseline_algorithm,x_train,y_train,x_test,y_test)
    