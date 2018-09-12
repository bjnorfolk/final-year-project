import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import io
from scipy.misc import imsave
from scipy import signal
from skimage import feature
from skimage import measure
from skimage import filters
import scipy.ndimage as nd
from skimage import morphology as morph
from skimage.morphology import square

sortedGlob = sorted(glob.glob("pdc/sdata/Vid89_good/*.bmp"))
images = [io.imread(file,as_gray=True) for file in sortedGlob]
img_array = np.array(images)
signma_c = 1
s_h = 2.5

a, b = 560, 450
n = 1112
m = 1024
r = 470

y,x = np.ogrid[-a:n-a, -b:m-b]
mask = x*x + y*y >= r*r

for index in range(img_array.shape[0]):

  r = 455
  mask = x*x + y*y >= r*r
  crop = img_array[index]
  crop[mask] = True
  hxx, hyy, hxy = hessian_matrix(crop, sigma=s_h, 
                                order='xy')
  i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)

  thresh=0.0012
  i_t2 = i2 > -thresh
  r = 435
  mask = x*x + y*y >= r*r
  i_t2[mask] = True
  mor = np.logical_not(i_t2)

  
  m9 = morph.erosion(mor)
  #m11 = morph.opening(mor, square(3))

  m6 = morph.remove_small_objects(m9, min_size=240, connectivity=30)
  m12 = morph.dilation(m6)
  m7 = morph.closing(m12)
  m7 = m7.astype(int)

  blobs = m7 > 1 * m7.mean()

  all_labels = measure.label(blobs)
  blobs_labels = measure.label(blobs, background=0)

  fig1, axe1 = plt.subplots(1, 5, figsize=(15, 6))
  ax1 = axe1.ravel()
  
  ax1[0].imshow(crop, cmap=cm.gray)
  ax1[0].set_title('Input Image')

  ax1[1].imshow(i2, cmap=cm.gray)
  ax1[1].set_title('Hessian sig='+str(s_h))

  ax1[2].imshow(mor, cmap=cm.gray)
  ax1[2].set_title('Threshold ='+str(thresh))

  ax1[3].imshow(m9, cmap=cm.gray)
  ax1[3].set_title('Morphology opening')

  ax1[4].imshow(m7, cmap=cm.gray)
  ax1[4].set_title('Morphology comb')
  

  plt.tight_layout()
  fig1.savefig('pdc/blob/' +str(index) +'.jpg')
  plt.close()

  fig2, axe2 = plt.subplots(1, 4, figsize=(15, 6))
  ax2 = axe2.ravel()

  ax2[0].imshow(crop, cmap=cm.gray)
  ax2[0].set_title('Input Image')

  ax2[1].imshow(m7, cmap=cm.gray)
  ax2[1].set_title('Morphology comb')

  ax2[2].imshow(all_labels, cmap='spectral')
  ax2[2].set_title('Blob')
 
  ax2[3].imshow(blobs_labels, cmap='spectral')
  ax2[3].set_title('Blob')

  plt.tight_layout()
  fig2.savefig('pdc/blob/' +'blob'+str(index) +'.jpg')
  plt.close()

  print(index)

  #20,000fps


