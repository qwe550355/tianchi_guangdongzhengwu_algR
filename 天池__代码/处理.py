import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tifffile as tiff
# x1 = plt.imread('result/result_predict_201711021516_3000.jpg')
# x2 = plt.imread('haha.jpg')
x2 = tiff.imread('E:\jinda\pycharm\FCN8d\\result/result_10310949.tif')
# x1 = np.load('result/result_predict_201711021516_3000.npy')
x1 = tiff.imread('good_result2.tif')
# plt.imshow(np.squeeze(x2,[2]))
# plt.colorbar()
# plt.show()
def theshold_value_cal(mat, theshold):
    mat_new = np.zeros((3000, 15106, 1), dtype=np.uint8)
    for i in range(3000):
        for j in range(15106):
            if mat[i][j][0] >= theshold:
                mat_new[i][j][0] = 1
            else:
                mat_new[i][j][0] = 0
    return mat_new
# x2 = theshold_value_cal(x2,128)
# x1 = theshold_value_cal(x1,128)
for i in range(3000):
    for j in range(15106):
        if x2[i][j][0] == 1:
            x1[i][j][0] = 1
plt.imshow(np.squeeze(x1,[2]))
plt.colorbar()
plt.show()
tiff.imsave('good_result3', x1)
# x1 = theshold_value_cal(x1, 128)
# x2 = theshold_value_cal(x2, 128)
# for i in range(0, 3000):
#     for j in range(0,960):
#         x1[i][j][0] = x2[i][j][0]
# for i in range(0, 3000):
#     for j in range(15106-768, 15106):
#         x1[i][j][0] = x2[i][j][0]
# np.save('result/result_predict_201711021516_3000.npy', x1)
# plt.imsave('result/solved_result.jpg', x1)