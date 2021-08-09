import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

def dist(a, b, axis=None):
    """ Return the Euclidean distance between points (can be vectors)"""
    return np.linalg.norm(a-b, axis=axis)

def nearest_cluster(point, clusters):
    """ Return the index and distance of the nearest cluster"""
    index = np.argmin(dist(point, clusters, axis=1))
    return index, dist(point, clusters[index])

def initialise_centroids(X, k):
    """ Randomly pick k pixels as centroids

    Output:
    C : array of pixel coordinates as centroids
    """
    C  = np.zeros([k,X.shape[2]], dtype=np.uint8)
    randx = np.random.choice(X.shape[0], k, replace=False)
    randy = np.random.choice(X.shape[1], k, replace=False)
    for i in range(k):
        C[i] = X[randx[i],randy[i]]
    return C

def initialise_centroids_kmeanspp(X, k):
    """ Initialise clusters using k-means++

    Output:
    C : array of centroids
    """
    # use a random data point as the first centroid
    randx = np.random.randint(0, X.shape[0])
    randy = np.random.randint(0, X.shape[1])
    C = np.array([X[randx,randy]])
    for i in range(1, k):
        sum = 0
        d = np.zeros((X.shape[0], X.shape[1]))
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                _, d[j, k] = nearest_cluster(X[j, k], C)
                d[j, k] *= d[j, k]
                sum += d[j, k]
        sum *= np.random.rand()
        added = False
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                sum -= d[j, k]
                if (sum <= 0):
                    C = np.append(C, X[j, k].reshape(1, X.shape[2]), axis=0)
                    added = True
                    break
            if added:
                break
    return C

def assignment_step(C, X):
    """ Calculate the distance between data points and centroids and pick the closest centroid.

    Output:
    S : array of integers to indicate which cluster every data point is assigned to
    """
    S = np.zeros((X.shape[0], X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            S[i][j], _ = nearest_cluster(X[i,j], C)
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

    C = initialise_centroids_kmeanspp(X, k)
    # Code for visualising initial centroids
    # for i in range(k):
    #     j, k = C[i][3], C[i][4]
    #     img_rgb[j-5:j+5, k-5:k+5] = [0, 255, 0]
    # plt.imshow(img_rgb)
    # plt.show()
    i = 100 # max iterations
    while(i>0):
        i -= 1
        print(100-i)
        S = assignment_step(C, X)
        C = update_step(C, X, S)
        new_S = assignment_step(C, X)
        similarity =  np.sum(np.equal(new_S, S)) / (X.shape[0] * X.shape[1])
        if(similarity > 0.99): # converge, return early
            return C, new_S
    return C, new_S

# %%
k = 5

img = cv2.imread('peppers.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#plt.savefig('../Report/figures/fig_2.2_mandm.eps', dpi=100, format='eps')
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# cols = np.zeros((img.shape[0], img.shape[1], 2), dtype=int)
# img = np.append(img, cols, axis=2)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img[i,j,3] = i
#         img[i,j,4] = j

# measure run time
import timeit
start = timeit.default_timer()
C, S = my_kmeans(img, k)
stop = timeit.default_timer()
#print(stop-start)

img = img[:,:,:3]
for i in range(k):
    img[np.where(S==i)] = C[i,:3]
img = img.astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
plt.imshow(img)
#plt.savefig('../Report/figures/fig_2.3_p_3d5.eps', dpi=100, format='eps')
plt.show()
