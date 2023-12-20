
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
    samples, labels = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    samples+= 2 * rng.uniform(size=samples.shape)
    linearly_separable = (samples,labels)
    return  [  make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
            ]

def train_classifiers(classifiers,datasets):
    """ train a seperate classifier for each dataset"""
    train_scores = []
    test_scores = []
    for cls,dataset in zip(classifiers,datasets):
        X,y = dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)
        cls.fit(X_train,y_train)
        prediction_x_test = cls.predict(X_test)
        prediction_x_train = cls.predict(X_train)
        x_test_score  = accuracy_score(y_test,prediction_x_test)
        x_train_score = accuracy_score(y_train,prediction_x_train)
        train_scores.append(x_train_score)
        test_scores.append(x_test_score)
    return train_scores,test_scores

def test_max_depths(datasets,classifiers):
    all_train_scores = []
    all_test_scores = []
    for current_max_depth,cls in enumerate(classifiers):
        classifiers = [ cls for _ in range(0,len(datasets))]
        #print(f"accuracy with max depth{current_max_depth}")
        train_scores,test_scores = train_classifiers(classifiers,datasets)
        all_train_scores.extend(train_scores)
        all_test_scores.extend(test_scores)    
    return all_train_scores,all_test_scores

def find_optimal_max_depth(test_accuracy_scores):
    """calculate the optimal maximum depth by 
    comparing the test accuracy scores
    for each dataset"""
    accuracy_moons,accuracy_circles,accuracy_linear = get_scores_for_datasets(test_accuracy_scores)
    print(f"accuracy moons:{max(accuracy_moons)}")
    print(f"accuracy circles:{max(accuracy_circles)}") 
    print(f"accuracy linear:{max(accuracy_linear)}")
    # add 1 because first entry tests with max_depth 1 
    max_depth_moons = np.argmax(accuracy_moons)+1
    max_depth_circles = np.argmax(accuracy_circles)+1
    max_depth_linear = np.argmax(accuracy_linear)+1
    return max_depth_moons,max_depth_circles,max_depth_linear

def get_scores_for_datasets(accuracy):
    return accuracy[0::3],accuracy[1::3],accuracy[2::3]

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
    plotted_datasets_name = [ "X_train with maximum depth","X_test with maximum depth",
                              "X_train with default values","X_test with default values",
                            ]
    dataset_count = 0
    for pair_of_cls,dataset in zip(classifiers,datasets):
        X,y = dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=0)
        xx,yy = create_meshgrid(X_train)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(3,5,position_idx)
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
            cls.fit(X_train,y_train)
            score = cls.score(X_test, y_test)
            for data,labels in zip([X_train,X_test],[y_train,y_test]):
                ax = plt.subplot(3,5,position_idx)
                Z= cls.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
                ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cm_bright,edgecolors='k')
                set_axis(ax,xx,yy)
                name = plotted_datasets_name[cls_name_idx]
                # first or second column or entire first row 
                if cls_name_idx in(0,1) or dataset_count==0:
                    ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')
                position_idx += 1
                cls_name_idx+=1
        dataset_count+=1



if __name__ =="__main__":
    datasets = generate_datasets()
    experimental_decision_trees = [DecisionTreeClassifier(max_depth=i) for i in range(1,15)]
    experimental_random_forests = [RandomForestClassifier(max_depth=i,random_state=42) for i in range(1,15)]
    train_acc_dct,test_acc_dct = test_max_depths(datasets,experimental_decision_trees)
    max_depth_moons_dct,max_depth_circles_dct,max_depth_linear_dct = find_optimal_max_depth(test_acc_dct)

    print("Maximum depth for Decision Trees:")
    print(f"moons:{max_depth_moons_dct} circles:{max_depth_circles_dct} linear:{max_depth_linear_dct}")
  

    optimal_decision_trees = [  DecisionTreeClassifier(max_depth=max_depth_moons_dct),
                                DecisionTreeClassifier(max_depth=max_depth_circles_dct),
                                DecisionTreeClassifier(max_depth=max_depth_linear_dct)]

    train_acc_rdf,test_acc_rdf = test_max_depths(datasets,experimental_random_forests)    
    max_depth_moons_rdf,max_depth_circles_rdf,max_depth_linear_rdf = find_optimal_max_depth(test_acc_rdf)
    print(f"Random Forest: maximum depths moons:{max_depth_moons_rdf}_rdf circles:{max_depth_circles_rdf} linear:{max_depth_linear_rdf}")
    optimal_random_forests = [  RandomForestClassifier(max_depth=max_depth_moons_rdf,random_state=42),
                                RandomForestClassifier(max_depth=max_depth_circles_rdf,random_state=42),
                                RandomForestClassifier(max_depth=max_depth_linear_rdf,random_state=42)]
    
    default_decision_trees = [DecisionTreeClassifier() for _ in range(0,3)]
    default_random_forest = [RandomForestClassifier() for _ in range(0,3)]
    decision_tree_clfs = list(zip(default_decision_trees,optimal_decision_trees))
    random_forest_clfs = list(zip(default_random_forest,optimal_random_forests))
    #plot_results(decision_tree_clfs,datasets)
    #plot_results(random_forest_clfs,datasets)
    #plot_tree(optimal_classifiers[0],filled=True)
    #plt.tight_layout()
    #plt.show() 
    