
###############################################
# ENGN4528 Clab2
# Ernest Kwan (u6381103)
# Task 1: Harris corner detector
###############################################

import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt


# %%
def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result

# %%
def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# %%
# Parameters, add more if needed
sigma = 2
thresh = 0.01
k = 0.04

img = cv2.imread('Harris_4.jpg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)

Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

# %%
######################################################################
# Task: Compute the Harris Cornerness
######################################################################
def cornerness(Ix2, Iy2, Ixy, k=0.04):
    """Compute the Harris cornerness given the image derivatives Ix2, Iy2 and Ixy

    Returns:
    R : the Harris cornerness for every pixel
    """
    det_M = Ix2 * Iy2 - Ixy * Ixy
    trace_M = Ix2 + Iy2
    R = det_M - k * (trace_M ** 2)
    return R

R = cornerness(Ix2, Iy2, Ixy, k)


# %%
######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
def thresholding_and_nms(R, thresh, neighbour_dist=3):
    """Apply thresholding on the Harris corners to only keep values above thresh * max value,
    and perform non-maximum suppression in 3x3 (default) neighbourhoods.

    Returns:
    out: a list of coordinates of the corners
    """
    thresh = thresh * R.max()
    r = neighbour_dist // 2
    out = []
    for i in range(r, R.shape[0]-r-1):
        for j in range(r, R.shape[1]-r-1):
            if R[i, j] > thresh and R[i-r:i+r+1, j-r:j+r+1].max() == R[i,j]:
                out.append([i, j])
    return out

corners = thresholding_and_nms(R, thresh, 3)
corners = np.array(corners)
img = cv2.imread('Harris_4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mark crosses for visualisation
for c in corners:
    img[c[0]-1, c[1]-1] = [0,255,0]
    img[c[0]-1, c[1]+1] = [0,255,0]
    img[c[0], c[1]] = [0,255,0]
    img[c[0]+1, c[1]+1] = [0,255,0]
    img[c[0]+1, c[1]-1] = [0,255,0]


plt.figure(figsize=(8,6))
plt.imshow(img)
#plt.savefig('../Report/figures/fig_1.4_my.eps', dpi=100, format='eps')
plt.show()

# Compare with opencv's built-in function
harris = cv2.cornerHarris(bw, 2, 3, k)
corners = thresholding_and_nms(harris, thresh, 3)
corners = np.array(corners)
img = cv2.imread('Harris_4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mark crosses for visualisation
for c in corners:
    img[c[0]-1, c[1]-1] = [0,255,0]
    img[c[0]-1, c[1]+1] = [0,255,0]
    img[c[0], c[1]] = [0,255,0]
    img[c[0]+1, c[1]+1] = [0,255,0]
    img[c[0]+1, c[1]-1] = [0,255,0]

plt.figure(figsize=(8,6))
plt.imshow(img)
#plt.savefig('../Report/figures/fig_1.4_cv2.eps', dpi=100, format='eps')
plt.show()

###############################################
# Task 2.1: my_kmeans() demo
###############################################
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


###############################################
# Task 2.2 - 2.3: k-means(++) for images
###############################################
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

###############################################
# Task 3: Eigenface face recognition
###############################################

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import os

def load_images_from_folder(folder):
    """ load all images in the folder
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return images

images = load_images_from_folder('trainingset')
images = np.array(images)
train_images = images
n, h, w = images.shape
d = h * w
X = np.reshape(images, (n, d)).copy() # each row represents an image

def plot_image(image):
    """ Visualise an image using matplotlib
    """
    image = np.reshape(image, (h, w))
    plt.figure(figsize=(4,4))
    plt.imshow(image, cmap='gray')
    #plt.savefig('../../Report/figures/fig_3.6_{}.eps'.format(i+1), dpi=100, format='eps')
    plt.show()

# %% Calculate the mean face
mean_face = np.mean(X, axis=0)
plot_image(mean_face)

# %% Subtract the mean face
A = X - mean_face

# %% Compute covariance matrix
cov_matrix = A @ A.T
cov_matrix /= n
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvectors = A.T @ eigenvectors
# sort in descending order
sort_indices = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:,sort_indices]
# Choose top k eigenvectors
k = 10
eigenvalues = eigenvalues[:k]
eigenvectors = eigenvectors[:,:k]
norms = np.linalg.norm(eigenvectors, axis=0)
eigenvectors /= norms
weights = eigenvectors.T @ X.T

# %% Visualise the top eigenfaces
for i in range(k):
    plot_image(np.real(eigenvectors.T[i]))

# %% Find the top 3 closest image to images in the test set

test_images = load_images_from_folder('testset')
# test_images = load_images_from_folder('mytestset') # Use my own images
test_images = np.array(test_images)
n, h, w = test_images.shape
d = w * h
test_images = np.reshape(test_images, (n, d))
i = 0
for image in test_images:
    plot_image(image)
    A_unknown = image - mean_face
    W_unknown = eigenvectors.T @ image
    diff = weights.T - W_unknown
    norms = np.linalg.norm(diff, axis=1)
    nearest = np.argsort(norms)[:3]
    plt.figure(figsize=(7,3))
    for j in range(3):
        image = train_images[nearest[j]]
        ax = plt.subplot(1, 3, j+1)
        ax.imshow(image, cmap='gray')
    #plt.savefig('../../Report/figures/fig_3.6_{}o.eps'.format(i+1), dpi=100, format='eps')
    plt.show()
    i += 1
    print('')
