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
res_x = 1280
res_y = 1024
img_b = 25
img_h = 10
noz_d = 6.2
#pixel per mm
# pixel_x = res_x/img_b
# pixel_y = res_y/img_h
# noz = pixel_y * noz_d

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

x0_points1 = []
y0_points1 = []
x1_points1 = []
y1_points1 = []

x0_points2 = []
y0_points2 = []
x1_points2 = []
y1_points2 = []

for line_ny in lines:
    ny0, ny1 = line_ny
    if ny1[0] > 1100 and ny0[0] > 1100 and ny1[1] > 600 and ny0[1] > 600:
     x0_points1.append(ny0[0])
     y0_points1.append(ny0[1])
     x1_points1.append(ny1[0])
     y1_points1.append(ny1[1])
    if ny1[0] > 1100 and ny0[0] > 1100 and ny1[1] < 600 and ny0[1] < 600:
     x0_points2.append(ny0[0])
     y0_points2.append(ny0[1])
     x1_points2.append(ny1[0])
     y1_points2.append(ny1[1])

x_points1 = x0_points1 + x1_points1
y_points1 = y0_points1 + y1_points1
x0 = sum(x_points1)/len(x_points1)
y0 = min(y_points1)
x1 = sum(x_points1)/len(x_points1)
y1 = max(y_points1)
ndy1 = y1 - y0
x_points2 = x0_points2 + x1_points2
y_points2 = y0_points2 + y1_points2
x0 = sum(x_points2)/len(x_points2)
y0 = min(y_points2)
x1 = sum(x_points2)/len(x_points2)
y1 = max(y_points2)
ndy2 = y1 - y0
noz_d = res_y - ndy2 - ndy1

x0_points = []
y0_points = []
x1_points = []
y1_points = []

for line_nx in lines:
    nx0, nx1 = line_nx
    if 1100 < nx1[0] < 1200 and 1100 < nx0[0] < 1200:
     if nx1[0] != nx0[0]:
      theta = np.arctan((nx1[1]-nx0[1])/(nx1[0]-nx0[0]))
      if (nx1[0]-nx0[0]) < 0 or (nx1[1]-nx0[1]) < 0:
       theta = theta + np.pi
       if 1.3 < theta < 2:
        x0_points.append(nx0[0])
        y0_points.append(nx0[1])
        x1_points.append(nx1[0])
        y1_points.append(nx1[1])
     else:
      x0_points.append(nx0[0])
      y0_points.append(nx0[1])
      x1_points.append(nx1[0])
      y1_points.append(nx1[1])

x_points = x0_points + x1_points
noz_x = sum(x_points)/len(x_points)

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

x0_points = []
y0_points = []
x1_points = []
y1_points = []

ax1[6].imshow(img, cmap=cm.gray)
for line_v in lines:
   v0, v1 = line_v
   if 10 < v1[0] < 1100 and 10 < v0[0] < 1100:
    if v1[0] != v0[0]:
     theta = np.arctan((v1[1]-v0[1])/(v1[0]-v0[0]))
     if (v1[0]-v0[0]) < 0 or (v1[1]-v0[1]) < 0:
      theta = theta + np.pi
     print(theta)
     if 1.3 < theta < 2:
      x0_points.append(v0[0])
      y0_points.append(v0[1])
      x1_points.append(v1[0])
      y1_points.append(v1[1])
      ax1[6].plot((v0[0], v1[0]), (v0[1], v1[1]), color='red', linewidth=2)
      print('graphed')
    else:
     ax1[6].plot((v0[0], v1[0]), (v0[1], v1[1]), color='red', linewidth=2)
     print('graphed1')
     #print(theta)
ax1[6].set_xlim((0, img.shape[1]))
ax1[6].set_ylim((img.shape[0], 0))
ax1[6].set_title('MD detection')
x_points = x0_points + x1_points
y_points = y0_points + y1_points
x0 = sum(x_points)/len(x_points)
y0 = min(y_points)
x1 = sum(x_points)/len(x_points)
y1 = max(y_points)
ax1[7].imshow(img, cmap=cm.gray)
ax1[7].plot((x0, x1), (y0, y1), color='red', linewidth=2)
plt.show()

l = y1 - y0
delta_x = noz_x - x1
delta_t = 1/1057
vel = delta_x/delta_t
non_d = l/noz_d
non_x = x1/noz_x

fig2, axe2 = plt.subplots(1, 2, figsize=(15, 6))
ax2 = axe2.ravel()

ax2[0].plot(non_d, n_x, color='red', linewidth=2)  
#ax2[1].plot(vel, n_x, color='red', linewidth=2) 