from base64 import standard_b64decode
from statistics import variance
from tkinter import E
from keras.datasets import mnist
import glob
from keras import backend as Keras
from keras.models import Sequential, load_model
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from module_ImageRecognize.transform_inputImg import transform_inputImg
from module_ImageRecognize.extract_frame import extract_frame

# filepath = "image_recognize/colab_mnist.hdf5"

# print("カレントパス", os.getcwd())
# print("filepath が指す絶対パス", os.path.abspath(filepath))
# print("ファイルが存在するかどうか", os.path.isfile(filepath))

# (train_data, train_teacher_labels), (test_data, test_teacher_labels) = mnist.load_data()

# BATHC_SIZE = 128
NUM_CLASSES = 10
# EPOCHS = 20

IMG_ROWS, IMG_COLS = 28, 28

# print('Channel 調整変換前 train_data shape:', train_data.shape)
# print('Channel 調整変換前 test_data shape:', test_data.shape)

# if Keras.image_data_format() == 'channel_first':
#     train_data = train_data.reshape(train_data.shape[0], 1, IMG_ROWS, IMG_COLS)
#     tesut_data = test_data.reshape(test_data.shape[0], 1, IMG_ROWS, IMG_COLS)
#     input_shape = (1, IMG_ROWS, IMG_COLS)
# else:
#     train_data = train_data.reshape(train_data.shape[0], IMG_ROWS, IMG_COLS, 1)
#     test_data = test_data.reshape(test_data.shape[0], IMG_ROWS, IMG_COLS, 1)
#     input_shape = (IMG_COLS, IMG_COLS, 1)

# print('Channel 調整変換後 train_data shape:', train_data.shape)
# print('Channel 調整変換後 test_data shape', test_data.shape)

# train_data = train_data.astype('float32')
# test_data = test_data.astype('float32')

# train_data /= 255
# test_data /= 255

keras_mnist_model = load_model('./image_recognize/colab_mnist.hdf5')

# prediction_array = keras_mnist_model.predict(test_data)

result_array = []
# files = glob.glob("./image_recognize/image/*")
files = glob.glob("./image_recognize/frame_red_image/*")
# files = ['./image_recognize/frame_red_image/WIN_20220329_10_43_11_Pro.jpg']
data_nums = len(files)
print(files)
for i in range(data_nums):
    print(files[i])
    moji_img = cv2.imread(files[i])

    # hsv_img = cv2.cvtColor(moji_img, cv2.COLOR_RGB2HSV)

    # array_h = []
    # array_s = []
    # array_v = []

    # sum = [0, 0, 0]
    # for a in range(hsv_img.shape[0]):
    #     for b in range(hsv_img.shape[1]):
    #         sum[0] = sum[0] + hsv_img[a][b][0]
    #         sum[1] = sum[1] + hsv_img[a][b][1]
    #         sum[2] = sum[2] + hsv_img[a][b][2]
    #         array_h.append(hsv_img[a][b][0])
    #         array_s.append(hsv_img[a][b][1])
    #         array_v.append(hsv_img[a][b][2])
    
    # average_hsv = [0, 0, 0]

    # average_hsv[0] = sum[0] / (hsv_img.shape[0] * hsv_img.shape[1])
    # average_hsv[1] = sum[1] / (hsv_img.shape[0] * hsv_img.shape[1])
    # average_hsv[2] = sum[2] / (hsv_img.shape[0] * hsv_img.shape[1])

    # std_d_h = np.std(array_h)
    # std_d_s = np.std(array_s)
    # std_d_v = np.std(array_v)

    # print("average_h:", average_hsv[0])
    # print("average_s:", average_hsv[1])
    # print("average_v:", average_hsv[2])

    # print("std_d_h:", std_d_h)
    # print("std_d_s:", std_d_s)
    # print("std_d_v:", std_d_v)

    extract_img = extract_frame(moji_img, files[i], None)

    # plt.imshow(moji_img)
    # plt.gray()
    # plt.show()

    WIDTH = 1280
    h, w = extract_img.shape[:2]
    HEIGHT = round(h * (WIDTH / w))
    extract_img = cv2.resize(extract_img, dsize=(WIDTH, HEIGHT))
    input_img = transform_inputImg(extract_img, files[i], IMG_ROWS, IMG_COLS)
    # print(input_img)

    handwritten_number_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    prediction_result = keras_mnist_model.predict(input_img)

    print(prediction_result[0])

    prediction_number = np.argmax(prediction_result[0])

    result_array.append(handwritten_number_names[prediction_number])

print(result_array)

# answer  = ['5', '3', '1', '1', '5', '3', '2', '4', '6', '8', 
#            '6', '1', '2', '2', '8', '0', '1', '2', '3', '4', 
#            '5', '6', '7', '8', '9', '1', '2', '3', '4', '5', 
#            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#            '2', '9', '4', '8', '5', '5', '3', '9', '6', '3']           
answer = ['3', '4', '9', '2', '4']


cnt = 0
for i in range(data_nums):
    if(result_array[i] == answer[i]):
        cnt += 1

correct_result_array = [[0] * NUM_CLASSES for i in range(NUM_CLASSES)]
for i in range(data_nums):
    correct_result_array[int(answer[i])][int(result_array[i])] += 1

for i in range(NUM_CLASSES):
    print(str(i) + ":", end='')
    print(correct_result_array[i])

print("精度:", cnt / data_nums)