import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

input2015 = 'label_final_none_final_2017_for_train_224.npy'
input2017 = 'label_2017_for_train_final_pic.npy'
data_2015 = np.load(input2015)
data_2017 = np.load(input2017)
print(data_2015.shape, data_2017.shape)
data_2015_with_2017 = np.concatenate((data_2015, data_2017), axis=0)
print(data_2015_with_2017.shape)
np.save('label_final_final_for_train_224.npy', data_2015_with_2017)

# plt.imshow(np.squeeze(data_2015_with_2017[1],2))
# plt.show()
# plt.imshow(np.squeeze(data_2015[1],2))
# plt.show()
