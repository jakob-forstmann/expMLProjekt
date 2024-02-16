import pandas as pd 
from sklearn.model_selection import GridSearchCV,cross_validate
from preprocessing import create_dataset,split_data

evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]

def evaluate_experiments(piped_model,custom_columns=None,dataset_loader=create_dataset):
    """evaluate the RMSE and MEA on the evaluation dataset using 5-cross validation
    returns the mean RMSE and MEA across 5 folds

    piped_model: a model stored in a sklearn.Pipeline, can be created 
    using the build_model function
    custom_columns: a list of columns to use,useful for testing different
    feature combinations"""
    spotify_songs = dataset_loader()
    if custom_columns is not None:
        spotify_songs = spotify_songs[custom_columns]
    X_train,_,y_train,_ = split_data(spotify_songs)
    cv_score = cross_validate(piped_model,X_train,y_train,scoring=evaluation_metrics,cv=5)
    full_results = pd.DataFrame(cv_score)
    return extract_RMSE_MEA_scores(full_results)

def perform_grid_search_cv(evaluation_parameter,piped_model,result_file_name:str):
    """" performs a grid search cv with the parameters stored in 
    evaluation_parameter.The results are saved in result_file_name.csv 
    due to potentially long execution times"""
    spotify_songs = create_dataset()
    X_train,_,y_train,_ = split_data(spotify_songs)
    scores = evaluation_parameter["evaluation_metrics"]
    param_to_test = evaluation_parameter["param_to_test"]
    cv_search = GridSearchCV(piped_model,param_to_test,n_jobs=-1,scoring=scores,
                                refit=False)
    cv_search.fit(X_train,y_train)
    df = pd.DataFrame(cv_search.cv_results_)
    columns_to_use = ["mean_test_neg_root_mean_squared_error","mean_test_neg_mean_absolute_error"] 
    model_parameters = ["param_"+val for val in evaluation_parameter["param_to_test"][0].keys()]
    columns_to_use.extend(model_parameters)
    reduced_df = df[columns_to_use]
    reduced_df.to_csv(result_file_name)


def extract_RMSE_MEA_scores(full_cross_val_result):
    """ returns the train and test RMSE and MEA mean across 5 folds"""
    columns_to_use = ["test_neg_root_mean_squared_error","test_neg_mean_absolute_error"] 
    cross_val_result = full_cross_val_result[columns_to_use]
    test_MEA_across_5_folds  = cross_val_result["test_neg_mean_absolute_error"].mean()
    test_RMSE_across_5_folds =  cross_val_result["test_neg_root_mean_squared_error"].mean()
    #print("test MEA std",cross_val_result["test_neg_mean_absolute_error"].std())
    #print("test RMSE std",cross_val_result["test_neg_root_mean_squared_error"].std())
    return test_RMSE_across_5_folds,test_MEA_across_5_folds

