import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tifffile as tif

x = tif.imread('result_final_logs_11111707_2500.tif')
plt.imshow(x[:,:,0])
plt.colorbar()
plt.show()