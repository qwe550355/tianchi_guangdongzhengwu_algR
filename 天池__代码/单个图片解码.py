import copy as cp
from collections import defaultdict
import csv
import sys
import cv2
import random as rd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from multiprocessing import Pool


def get_mat_from_try(data):
    mat = np.zeros((960, 960, 1))
    mat_trans = np.zeros((1120, 1120, 1))
    for i in range(1120 // 224):
        for j in range(1120 // 224):
            mat_trans[i*224:(i+1)*224, j*224:(j+1)*224] = data[i*(1120//224) + j]
    for i in range(960):
        for j in range(960):
            mat[i][j] = mat_trans[i][j]
    return mat


def get_mat_from_data(data):
    mat = np.zeros((4000, 15106, 1))
    mat_trans = np.zeros((4928, 15456, 1))
    for i in range(4928 // 224):
        for j in range(15456 // 224):
            mat_trans[i*224:(i+1)*224, j*224:(j+1)*224] = data[i*(15456//224) + j]
    for i in range(4000):
        for j in range(15106):
            mat[i][j] = mat_trans[i][j]
    return mat


def np_dec(np1, np2):
    high, width = np1.shape[:2]
    mat = np.zeros((high*width, 1), dtype=np.uint8)
    np1 = np.reshape(np1, (high*width))
    np2 = np.reshape(np2, (high*width))
    mat[np.min((np1 == 1, np2 == 0), axis=0)] = 1
    mat = np.reshape(mat, (high, width, 1))
    return mat


def np_add(np1, np2):
    high, width = np1.shape[:2]
    mat = np.zeros((high*width, 1), dtype=np.uint8)
    np1 = np.reshape(np1, (high*width))
    np2 = np.reshape(np2, (high*width))
    mat[np.max((np1 == 1, np2 == 1), axis=0)] = 1
    mat = np.reshape(mat, (high, width, 1))
    return mat


def theshold_value_cal(mat, theshold):
    high, width = mat.shape[:2]
    new_mat = np.zeros((high*width, 1), dtype=np.uint8)
    new_mat[np.reshape(mat, (high*width, 1)) > theshold] = 1
    new_mat = np.reshape(new_mat, (high, width, 1))
    return new_mat


def pic_unlock(url,outputname):
    x = np.load(url)
    t = get_mat_from_data(x)
    plt.imshow(np.squeeze(t,[2]))
    plt.colorbar()
    plt.show()
    np.save('result/'+str(outputname)+'.npy', t)
    plt.imsave('result/'+str(outputname)+'.jpg', np.squeeze(t,[2]))


def pic_unlock_try(url,outputname):
    x = np.load(url)
    t = get_mat_from_try(x)
    plt.imshow(np.squeeze(t,[2]))
    plt.colorbar()
    plt.show()
    np.save('result/'+str(outputname)+'.npy', t)
    plt.imsave('result/'+str(outputname)+'.jpg', np.squeeze(t,[2]))


def add_pic(url1, url2, outname):
    data_2017 = plt.imread(url1)
    data_2015 = plt.imread(url2)
    data_2015 = theshold_value_cal(data_2015[:, :, 0], 128)
    data_2017 = theshold_value_cal(data_2017[:, :, 0], 128)
    result = np_add(data_2015,data_2017)
    plt.imshow(np.squeeze(result, [2]))
    plt.colorbar()
    plt.show()
    tiff.imsave('result/' + outname + '.tif', result)
    plt.imsave('result/' + outname + '.jpg', np.squeeze(result, [2]))


def dec_pic(url1, url2, outname):
    data_2017 = plt.imread(url1)
    data_2015 = plt.imread(url2)
    plt.imshow(np.squeeze(data_2015,[2]))
    plt.colorbar()
    plt.show()
    data_2015 = theshold_value_cal(data_2015[:, :, 0], 128)
    data_2017 = theshold_value_cal(data_2017[:, :, 0], 128)
    result = np_dec(data_2017, data_2015)
    plt.imshow(np.squeeze(result,[2]))
    plt.colorbar()
    plt.show()
    tiff.imsave('result/' + outname+'.tif', result)
    plt.imsave('result/' + outname+'.jpg', np.squeeze(result, [2]))


def pic2tiff(url, outname, th_value, whethernp=1):
    if whethernp == 1:
        x = np.load(url)
    else:
        x = plt.imread(url)
    plt.imshow(np.squeeze(x, [2]))
    plt.colorbar()
    plt.show()
    print(x.shape)
    t = theshold_value_cal(np.squeeze(x[:, :, 0]), th_value)
    plt.imshow(np.squeeze(t, [2]))
    plt.colorbar()
    plt.show()
    tiff.imsave(outname, t)

# add_pic('result/result_final_logs_11091446_3000.jpg', 'result/result_final_logs_11111707_3000.jpg', outname='result_final_logs_11112246')
# pic_unlock_try('rt/resultnow.npy','old')
# pic_unlock('predict/result2017_logs_11111657.npy', 'result2017_logs_11111657')
# dec_pic('merge_label_2017_good.jpg', 'merge_label_change_2015.jpg', outname='answer')
pic2tiff('result_final_logs_11111707_2500_change.jpg', 'result_final_logs_11111707_2500.tif', 128, 0)






