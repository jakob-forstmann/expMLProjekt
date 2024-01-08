from models.baselines import Estimator 
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from preprocessing import create_dataset,split_data
from models.tree_models import Decision_Tree_Wrapper


def evaluate(model:Estimator,X_train,y_train):
    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error"]
    cv_score = cross_validate(model,X_train,y_train,scoring=evaluation_metrics,cv=5,error_score="raise")
    print(f"Evaulating with model {str(model)}")
    print(f"Model{model}: root mean squarred error",cv_score["test_neg_root_mean_squared_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())
    print(f"Model{model}: mean absolute error",cv_score["test_neg_mean_absolute_error"])
    print("mean across 5 folds",cv_score["test_neg_mean_absolute_error"].mean())


def train(model):
    spotify_songs = create_dataset()
    # to speedup evaluation time drop column track_album_name for now 
    spotify_songs = spotify_songs.drop(columns="track_album_name")
    if "track_album_name" in spotify_songs:
        vectorizer = ColumnTransformer(
            [("TF-IDF",TfidfVectorizer(),"track_album_name")],remainder="passthrough")
        model = make_pipeline(vectorizer,model)
    x_train,_,y_train,_ = split_data(spotify_songs)
    evaluate(model,x_train,y_train)       

def test_different_Decision_Trees():
    default_tree = Decision_Tree_Wrapper().get_model()
    train(default_tree)

if __name__ =="__main__":
    test_different_Decision_Trees()
