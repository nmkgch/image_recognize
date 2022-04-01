from base64 import standard_b64decode
from statistics import variance
from keras.datasets import mnist
import glob
from keras import backend as Keras
from keras.models import Sequential, load_model
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from module_ImageRecognize import transform_inputImg

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

def std_divi(array, ave):
    square_total = 0
    cnt_not_zero = 0
    for i in range(IMG_ROWS):
        for j in range(IMG_COLS):
            if(array[i][j] != 0):
                cnt_not_zero += 1
                square_total += (array[i][j] - ave) ** 2
    vari = square_total / cnt_not_zero
    standard_diviation = math.sqrt(vari)
    return standard_diviation

result_array = []
# files = glob.glob("./image_recognize/image/*")
files = ['./image_recognize/character0.png']
data_nums = len(files)
print(files)
for i in range(data_nums):
    print(files[i])
    moji_img = cv2.imread(files[i])

    # plt.imshow(moji_img)
    # plt.gray()
    # plt.show()

    WIDTH = 1280
    h, w = moji_img.shape[:2]
    HEIGHT = round(h * (WIDTH / w))
    moji_img = cv2.resize(moji_img, dsize=(WIDTH, HEIGHT))
    input_img = transform_inputImg.transform_inputImg(moji_img, files[i], IMG_ROWS, IMG_COLS)
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
answer = ['3']


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