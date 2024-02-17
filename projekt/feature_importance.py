import pandas as pd 
from preprocessing import create_dataset,split_data,build_model
from find_DT_parameters import get_optimized_dt
from find_lasso_parameters import get_optimized_linear_model
from find_RDF_parameters import get_optimized_rdf
from dataset_statistics import create_bar_plot


def calculate_feature_importance(model):
    """ calculates the feature importance for each column
    returns the feature importance for all columns and for 
    all columns except track name"""
    spotify_songs = create_dataset()
    piped_model = build_model(model)
    x_train,_,y_train,_ = split_data(spotify_songs)
    piped_model.fit(x_train,y_train)
    fitted_model = piped_model[1]
    vect = piped_model[0]
    feature_names = vect.get_feature_names_out()
    column_names  = list(map(lambda val:val.replace("remainder__","").replace("TF-IDF__",""),feature_names))
    feat_importance= fitted_model.feature_importances_   
    words_importance = pd.Series(feat_importance[:-4],index=column_names[:-4])
    column_importance = pd.Series(feat_importance[-4:],index=column_names[-4:])
    return column_importance,words_importance


def plot_top_twenty_words(words_importance):
    tf_idf_values = words_importance.sort_values(ascending=False)[0:20]
    create_bar_plot(tf_idf_values,"word","importance")

if __name__== "__main__":
    opt_dt = get_optimized_dt()
    opt_linear_model = get_optimized_linear_model()
    opt_rdf = get_optimized_rdf()
    columns_importance_dt,words_importance_dt = calculate_feature_importance(opt_dt)
    plot_top_twenty_words(words_importance_dt)
    columns_importance_rdf,words_importance_rdf = calculate_feature_importance(opt_rdf)
    columns_importance = pd.DataFrame({"DT":columns_importance_dt,"RDF":columns_importance_rdf})
    plot_top_twenty_words(words_importance_rdf)
    create_bar_plot(columns_importance,"column","feature importance")
