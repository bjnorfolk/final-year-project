import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage import io
from scipy.misc import imsave
from scipy import signal
from skimage import feature
from skimage import measure
from skimage import filters
import scipy.ndimage as nd
from skimage import morphology as morph
from skimage.morphology import square
from skimage.filters import gaussian
from skimage.morphology import thin

#"pdc/sdata/Vid89_good/vid_089_000050.bmp"
file = "pdc/st_2018/1057fps_t2c/t20029.tif"
img = io.imread(file)

signma_c = 1
s_h = 2.5
s_g = 2.5
t_percent = 90
thinning = 1

#circular mask
# a, b = 560, 450
# n = 1112
# m = 1024
# r = 470

# y,x = np.ogrid[-a:n-a, -b:m-b]
# mask = x*x + y*y >= r*r

hxx, hyy, hxy = hessian_matrix(img, sigma=s_h, 
                                order='xy')
i_h1, i_h2 = hessian_matrix_eigvals(hxx, hxy, hyy)
i_g = gaussian(i_h1, sigma=s_g)
i_e = morph.erosion(i_g)
i_t = np.percentile(i_e[i_e > 0], t_percent)
i_b = i_e > i_t
# r = 435
# mask = x*x + y*y >= r*r
# i_b[mask] = False
i_rs = morph.remove_small_objects(i_b, min_size=240, connectivity=30)
i_th = thin(i_rs, thinning)
blobs = i_th > 1 * i_th.mean()
all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0)

i_f = i_th

lines = probabilistic_hough_line(i_f, threshold=10, line_length=150,
                                 line_gap=10)
lines_v = lines

fig1, axe1 = plt.subplots(2, 4, figsize=(15, 6))
ax1 = axe1.ravel()

ax1[0].imshow(img, cmap=cm.gray)
ax1[0].set_title('Input Image')
ax1[1].imshow(i_h1, cmap=cm.gray)
ax1[1].set_title('Hessian sig='+str(s_h))
ax1[2].imshow(i_e, cmap=cm.gray)
ax1[2].set_title('Gaussian sig='+str(s_g))
ax1[3].imshow(i_b, cmap=cm.gray)
ax1[3].set_title('Threshold %='+str(t_percent))
ax1[4].imshow(blobs_labels, cmap='Spectral')
ax1[4].set_title('CC')

ax1[5].imshow(img, cmap=cm.gray)
for line in lines:
    p0, p1 = line
    if p1[0] != p0[0]:
     theta = np.arctan((p1[1]-p0[1])/(p1[0]-p0[0]))
    ax1[5].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax1[5].set_xlim((0, img.shape[1]))
ax1[5].set_ylim((img.shape[0], 0))
ax1[5].set_title('Probabilistic Hough')

ax1[6].imshow(img, cmap=cm.gray)
for line_v in lines_v:
   v0, v1 = line_v
   if 10 < v1[0] < 1100 and 10 < v0[0] < 1100:
    if v1[0] != v0[0]:
     theta = np.arctan((v1[1]-v0[1])/(v1[0]-v0[0]))
     if (v1[0]-v0[0]) < 0:
      theta = theta + np.pi
     if 1.3 < int(theta) < 2:
      ax1[6].plot((v0[0], v1[0]), (v0[1], v1[1]), color='red', linewidth=2)
      print('graphed')
    else:
     ax1[6].plot((v0[0], v1[0]), (v0[1], v1[1]), color='red', linewidth=2)
     print('graphed1')
     #print(theta)
ax1[6].set_xlim((0, img.shape[1]))
ax1[6].set_ylim((img.shape[0], 0))
ax1[6].set_title('MD detection')

plt.show()


  
