import numpy as np
import scipy.ndimage as ndi
from skimage import data,filters,segmentation,measure,morphology,color
import matplotlib.pyplot as plt
from skimage import measure as mes
import numpy as np
import scipy.ndimage as ndi

import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tiff

x = plt.imread('result/result_final_logs_11111707_3000.jpg')

image = color.rgb2gray(x[:,:,0])
thresh = filters.threshold_otsu(image)
bw = morphology.closing(image > thresh, morphology.square(3))
#
# labels = measure.label(bw)
# dst = color.lab2rgb(labels)
# plt.imshow(bw)
# plt.colorbar()
# plt.show()

label_img = measure.label(bw, connectivity=1)
props = measure.regionprops(label_image=label_img)


f = open('data.txt', 'w')
for i in range(len(props)):
    txt = '质心坐标：' + str(props[i].centroid) + '面积:' + str(props[i].area) + '\n'
    f.write(txt)
f.close()