
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification

def generate_datasets():  
    """
    generates three different datasets which are splitted in samples and labels.
    returns:
        a tuple of (X_train,y_train),each containing a list of the samples 
        respective the labels for the datasets at the corresponding index
    """
    samples, labels = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    samples+= 2 * rng.uniform(size=samples.shape)
    linearly_separable = (samples,labels)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]
    X_train = [ sample for sample in datasets[0]]
    y_train = [ label  for label in datasets[1] ]
    
    return (X_train,y_train)


if __name__ =="__main__":
    generate_datasets()