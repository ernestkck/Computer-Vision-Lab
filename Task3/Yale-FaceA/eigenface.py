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
