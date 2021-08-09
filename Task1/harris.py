# %%
"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Ernest Kwan (u6381103)
"""

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
