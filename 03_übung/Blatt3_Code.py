
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generate_datasets():
    """generates three different datasets containing samples and labels zero or one"""
    samples, labels = make_classification(n_samples=200,n_features=2, n_redundant=0, n_informative=2,
                            random_state=42, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    samples+= 2 * rng.uniform(size=samples.shape)
    linearly_separable = (samples,labels)
    return  [  make_moons(n_samples=200,noise=0.3, random_state=42),
                make_circles(n_samples=200,noise=0.2, factor=0.5, random_state=42),
                linearly_separable
            ]

def train_classifiers(classifiers,datasets):
    for cls,dataset in zip(classifiers,datasets):
        X,y = dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)
        cls.fit(X_train,y_train)
        prediction_x_test = cls.predict(X_test)
        prediction_x_train = cls.predict(X_train)
        x_test_score  = accuracy_score(y_test,prediction_x_test)
        x_train_score = accuracy_score(y_train,prediction_x_train)
        print(f"test score: {x_test_score} train score: {x_train_score}")
    return classifiers


def create_meshgrid(x_train):
    """ creates a meshgrid from the training dataset"""
    h = .02  # step size in the mesh
    x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
    y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
    return np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

def set_axis(ax,x,y):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
def plot_results(classifiers,datasets):
    figure = plt.figure(figsize=(27, 11)) 
    position_idx =1
    plotted_datasets_name = [ "X_train with maxium depth","X_test with maxium depth",
                              "X_train with default values","X_test with default values",
                              "X_train with a random forest","X_test with a random forest"]
    dataset_count = 0
    for pair_of_cls,dataset in zip(classifiers,datasets):
        X,y = dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)
        xx,yy = create_meshgrid(X_train)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(3,7,position_idx)
        if position_idx == 1:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
        set_axis(ax,xx,yy)
        position_idx+=1
        cls_name_idx = 0
        for cls in pair_of_cls:
            for data,labels in zip([X_train,X_test],[y_train,y_test]):
                ax = plt.subplot(3,7,position_idx)
                Z= cls.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
                ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cm_bright,edgecolors='k')
                set_axis(ax,xx,yy)
                name = plotted_datasets_name[cls_name_idx]
                if cls_name_idx in (0,1):
                    name+=" "+str(cls.tree_.max_depth)
                # first or second column or entire first row 
                if cls_name_idx in(0,1) or dataset_count==0:
                    ax.set_title(name)
                position_idx += 1
                cls_name_idx+=1
        dataset_count+=1

if __name__ =="__main__":
    datasets = generate_datasets()
    optimal_classifiers = [ DecisionTreeClassifier(max_depth=5),DecisionTreeClassifier(max_depth=4),DecisionTreeClassifier(max_depth=2)]
    print("optimal classifiers")
    optimal_classifiers = train_classifiers(optimal_classifiers,datasets)
    print("default classifiers")
    default_classifiers = [DecisionTreeClassifier() for _ in range(0,3)]
    default_classifiers = train_classifiers(default_classifiers,datasets)
    random_forest = [RandomForestClassifier() for _ in range(0,3)]
    print("random forest classifier")
    train_classifiers(random_forest,datasets)
    combined_classifiers =list(zip(optimal_classifiers,default_classifiers,random_forest))
    plot_results(combined_classifiers,datasets)
    #plot_tree(optimal_classifiers[0],filled=True)
    plt.tight_layout()
    plt.show() 
