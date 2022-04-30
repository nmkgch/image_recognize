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
from PIL import Image
from PIL.ExifTags import TAGS
from module_ImageRecognize.transform_inputImg import transform_inputImg
from module_ImageRecognize.extract_frame import extract_frame
from module_ImageRecognize.select_image_file import select_image_file

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
whole_result_array = []
# files = glob.glob("./image_recognize/image/*")
# files = glob.glob("./image_recognize/frame_red_image/*")
# files = ['./image_recognize/frame_point_image/WIN_20220430_23_26_40_Pro.jpg']
files = [select_image_file()]
image_nums = len(files)
character_nums = 0
print(files)
for i in range(image_nums):
    print(files[i])
    #入力はRGB形式
    img = Image.open(files[i])

    #exif情報を取得
    exif = img._getexif()

    #exif情報からOrientationを取得
    exif_data = []
    for id, value in exif.items():
        if TAGS.get(id) == 'Orientation':
            tag = TAGS.get(id, id),value
            exif_data.extend(tag)

    if exif_data[1] == 3:
        #180度回転
        img = img.transpose(Image.ROTATE_180)
    elif exif_data[1] == 6:
        #270度回転
        img = img.transpose(Image.ROTATE_270)
    elif exif_data[1] == 8:
        #90度回転
        img = img.transpose(Image.ROTATE_90)

    img = np.array(img)
    #RGB形式をBGR形式に変換(OpenCVが基本的にBGRなため)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    WIDTH = 1280
    h, w = img.shape[:2]
    HEIGHT = round(h * (WIDTH / w))
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))

    character_img_array, frame_number = extract_frame(img, None)
    
    result_array = [None] * len(character_img_array)
    for index in range(len(character_img_array)):
        character_nums += 1

        character_img = character_img_array[index]

        # plt.imshow(moji_img)
        # plt.gray(

        input_img = transform_inputImg(character_img, IMG_ROWS, IMG_COLS)
        # print(input_img)

        handwritten_number_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        prediction_result = keras_mnist_model.predict(input_img)

        print(prediction_result[0])

        prediction_number = np.argmax(prediction_result[0])

        result_array[frame_number[index]] = handwritten_number_names[prediction_number]

    whole_result_array.append(result_array)

print(whole_result_array)

# answer  = ['5', '3', '1', '1', '5', '3', '2', '4', '6', '8', 
#            '6', '1', '2', '2', '8', '0', '1', '2', '3', '4', 
#            '5', '6', '7', '8', '9', '1', '2', '3', '4', '5', 
#            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#            '2', '9', '4', '8', '5', '5', '3', '9', '6', '3']           
# answer = ['3', '4', '9', '2', '4']
answer = [['6', '3', '2', '4', '1']]


cnt = 0
for i in range(image_nums):
    for j in range(len(whole_result_array[i])):
        if(whole_result_array[i][j] == answer[i][j]):
            cnt += 1

correct_result_array = [[0] * NUM_CLASSES for i in range(NUM_CLASSES)]
for i in range(image_nums):
    for j in range(len(whole_result_array[i])):
        correct_result_array[int(answer[i][j])][int(whole_result_array[i][j])] += 1

for i in range(NUM_CLASSES):
    print(str(i) + ":", end='')
    print(correct_result_array[i])

print("精度:", cnt / character_nums)