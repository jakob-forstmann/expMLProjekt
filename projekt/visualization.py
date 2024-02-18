from preprocessing import build_model,create_dataset,split_data
from find_DT_parameters import get_optimized_dt
from find_RDF_parameters import get_optimized_rdf
from find_lasso_parameters import get_optimized_linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay

def reduce_dimensionality(dataset):
    """ reduces the spotify dataset to two 
        dimensions using Truncated SVD """
    # create a pipeline with just the tf-idf vectorizer
    tf_idf_pipeline = build_model(None)[0]
    transformed_data = tf_idf_pipeline.fit_transform(dataset)
    pca = TruncatedSVD(n_components=2)
    return pca.fit_transform(transformed_data)


def plot_decision_boundaries(regressor):
    """ plots the decision boundary for the regressor 
        in 2D using the reduced spotify dataset.""" 
    spotify_songs = create_dataset()
    complete_dataset_reduced = reduce_dimensionality(spotify_songs)
    X_train,X_test,y_train,y_test = split_data(spotify_songs)
    X_train_reduced = reduce_dimensionality(X_train)
    X_test_reduced = reduce_dimensionality(X_test)
    regressor.fit(X_train_reduced,y_train)
    x_label = "Truncated SVD Dimension 1"
    y_label = "Truncated SVD Dimension 2"
    boundary  = DecisionBoundaryDisplay.from_estimator(regressor,complete_dataset_reduced,xlabel=x_label,ylabel=y_label)
    x_test_scatter = boundary.ax_.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1],c=y_test, cmap="magma",edgecolors='k')
    boundary.ax_.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1],c=y_train, alpha=0.6,cmap="magma",edgecolors='k')
    boundary.figure_.colorbar(x_test_scatter).set_label("valence")
    plt.show()


def plot_linear_reg_residuals(regressor):
    spotify_songs = create_dataset()
    piped_liner = build_model(regressor)
    X_train,X_test,y_train,y_test = split_data(spotify_songs)
    piped_liner.fit(X_train,y_train)
    y_pred = piped_liner.predict(X_test)
    PredictionErrorDisplay.from_predictions(y_true=y_test,y_pred=y_pred)  
    plt.show()

def plot_dataset_2D(data,target):
    """ plots the reduced dataset in 2D encoding the
        corresponding valence range with a color range"""
    plt.scatter(data[:,0],data[:,1],c=target,cmap="magma")
    plt.title("Dataset reduced to 2 Dimension")
    plt.xlabel("Truncated SVD Dimension 1")
    plt.ylabel("Truncated SVD Dimension 2")
    plt.colorbar().set_label("valence")
    plt.show()

def plot_Dataset_3D(data,target):
    """ plots the reduced dataset in 3D"""
    fig = plt.figure(1, figsize=(18, 12))
    plt.clf()
    ax = fig.add_subplot(111, projection="3d",elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    ax.scatter(data[:, 0],data[:, 1],target,edgecolor="k")
    ax.set_zlabel("valence")
    ax.set_xlabel("Truncated SVD Dimension 1")
    ax.set_ylabel(" Trundcated SVD Dimension 2")
    plt.title("Datast reduced to 2 dimensions")
    plt.show()


def plot_predictions(model):
    """ creates a scatter plot comparing the predicted 
        valence values for the test Datenset with the true 
        valence values. """
    piped_model = build_model(model)
    spotify_songs = create_dataset()
    X_train,X_test,y_train,y_test = split_data(spotify_songs)
    piped_model.fit(X_train,y_train)
    pred = piped_model.predict(X_test)
    plt.scatter(range(len(y_test)),y_test,label='true values', color='blue')
    plt.scatter(range(len(y_test)),pred,label='predicted values', color='red')
    plt.title('Comparison of true vs predicted Values')
    plt.xlabel('Testing Datapoints')
    plt.ylabel('valence ')
    plt.legend()
    plt.show()



if __name__ =="__main__":
    opt_rdf = get_optimized_rdf()
    opt_dt = get_optimized_dt()
    opt_linear_modell = get_optimized_linear_model()
    plot_linear_reg_residuals(opt_linear_modell)
    plot_linear_regression(opt_dt)
    plot_decision_boundaries(opt_dt)
    plot_decision_boundaries(opt_rdf)
    plot_predictions(opt_linear_modell)
    plot_predictions(opt_dt)
    plot_predictions(opt_rdf)

