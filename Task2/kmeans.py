import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    """ Generates data points with 4 Gaussian distributions

    Output:
    X : 400 by 2 array of floats
    """
    X = np.zeros((400, 2))
    mean = np.array([0.5, 0.5])
    cov = np.identity(2)
    cov *= 1/30
    X[:100] = np.random.multivariate_normal(mean, cov, 100)
    mean = np.array([-0.5, 0.5])
    X[100:200] = np.random.multivariate_normal(mean, cov, 100)
    mean = np.array([0.75, -0.5])
    X[200:300] = np.random.multivariate_normal(mean, cov, 100)
    mean = np.array([-0.25, -0.5])
    X[300:] = np.random.multivariate_normal(mean, cov, 100)
    return X

def initialise_centroids(X, k):
    """ Randomly pick k data points as centroids

    Output:
    C : array of floats as centroids
    """
    C  = np.zeros([k,2])
    rand = np.random.choice(X.shape[0], k, replace=False)
    for i in range(k):
        C[i] = X[rand[i]]
    return C

def assignment_step(C, X):
    """ Calculate the distance between data points and centroids and pick the closest centroid.

    Output:
    S : array of integers to indicate which cluster every data point is assigned to
    """
    S = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        dist = np.linalg.norm(X[i]-C, axis=1)
        S[i] = np.argmin(dist)
    return S

def update_step(C, X, S):
    """ Update the centroids with the mean of the assigned data points

    Output:
    C : array of floats
    """
    C = np.zeros(C.shape)
    for i in range(C.shape[0]):
        C[i] = X[np.where(S==i)].mean(axis=0)
    return C

def my_kmeans(X, k):
    """
    Inputs:
    X : array of data points to be processed
    k : number of clusters

    Output:
    C : centroids of the clusters
    new_S : the clusters assigned to every data point
    """

    C = initialise_centroids(X, k)
    # for i in range(k):
    #     plt.scatter(C[i,0], C[i,1], marker='*', s=1000, c=colours[i])
    # plt.savefig('../Report/figures/fig_2.1a.eps', dpi=100, format='eps')
    # plt.show()
    i = 50 # max iterations
    while(i>0):
        i -= 1
        S = assignment_step(C, X)
        C = update_step(C, X, S)
        new_S = assignment_step(C, X)
        if(np.array_equal(new_S, S)): # converge, return early
            return C, new_S
    return C, new_S

X = generate_data()
# plt.scatter(X[:,0], X[:,1], s=12, c='dimgray')


k = 4
C, S = my_kmeans(X, k)
colours = ['r', 'g', 'b', 'y']

for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], s=12, c=colours[S[i]])
for i in range(k):
    plt.scatter(C[i,0], C[i,1], marker='*', s=1000, c=colours[i])
# plt.savefig('../Report/figures/fig_2.1b.eps', dpi=100, format='eps')
plt.show()
