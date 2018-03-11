import numpy as np
import scipy.ndimage as ndi
from skimage import data,filters,segmentation,measure,morphology,color
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff

x = plt.imread('result_final_logs_11111707_2500_change.bmp')

# x = tiff.imread('result/final_result_pred_.tif')
# plt.imshow(np.squeeze(x, [2]))
# plt.colorbar()
# plt.show()

# for i in range(0, 1920):
#     for j in range(15106-728, 15106):
#         x[i][j][0] = 0
#
# for i in range(3577, 4000):
#     for j in range(8300, 8800):
#         x[i][j][0] = 0

print(x.shape)
# plt.imshow(np.squeeze(x, [2]))
# plt.colorbar()
# plt.show()
# t = np.zeros((3000,15106))
# t = x[:3000, :]
# cv.imwrite('change.jpg', t)

image = color.rgb2gray(x[:,:,0])
thresh = filters.threshold_otsu(image)
bw = morphology.closing(image > thresh, morphology.square(3))


# plt.imshow(bw)
# plt.colorbar()
# plt.show()
# plt.imsave('result/unchange2015.jpg', bw)
dst = morphology.remove_small_objects(bw, min_size=50, connectivity=1)

# plt.imshow(dst)
# plt.colorbar()
# plt.show()
plt.imsave('result_final_logs_11111707_2500_change.jpg', dst)
# tiff.imsave('result/final_result_solved_50.tif', dst)
