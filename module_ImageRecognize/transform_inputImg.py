import cv2
import numpy as np
import os
import pathlib
from keras import backend as Keras
import matplotlib.pyplot as plt

def transform_inputImg(img, IMG_ROWS, IMG_COLS):
    subfig = []
    fig = plt.figure(figsize=(6.4, 2.4))
    PLOT_ROWS = 1
    PLOT_COLUMNS = 3

    subfig.append(fig.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 1))
    subfig[0].imshow(img)

    # dilate_var = 50
    # kernel = np.ones((dilate_var, dilate_var), np.uint8)
    # img = cv2.dilate(img, kernel, iterations = 1)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    a = (0, 0, 0)
    b = (180, 80, 130)
    result_img = cv2.inRange(hsv_img, a, b)

    close_var = 1
    kernel = np.ones((close_var, close_var), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, kernel)

    open_var = 1
    kernel = np.ones((open_var, open_var), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_OPEN, kernel)

    # plt.imshow(result_img)
    # plt.show()

    # file_name, ext = os.path.splitext( os.path.basename(file_path) )
    # new_path = './image_recognize/image_io/' + file_name + '_io' + ext

    # new_file = pathlib.Path(new_path)
    # new_file.touch()

    # cv2.imwrite(new_path, result_img)

    subfig.append(fig.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 2))
    subfig[1].imshow(result_img)

    right = 0
    left = result_img.shape[1] - 1
    down = 0
    up = result_img.shape[0] - 1
    for i in range(result_img.shape[0]):
        for j in range(result_img.shape[1]):
            if(result_img[i][j] == 0):
                if(right < j):
                    right = j
                if(left > j):
                    left = j
                if(down < i):
                    down = i
                if(up > i):
                    up = i
    if(left - right > down - up):
        dif = (right - left) - (down - up)
        up = up - int(dif / 2)
        down = down + int(dif / 2)
    else:
        dif = (down - up) - (right - up)
        left = left - int(dif / 2)
        right = right + int(dif / 2)

    space_bit = 20
    if(up - space_bit >= 0):
        up -= space_bit
    else:
        up = 0
    if(down + space_bit <= result_img.shape[0] - 1):
        down += space_bit
    else:
        down = result_img.shape[0] - 1
    if(left - space_bit >= 0):
        left -= space_bit
    else:
        left = 0
    if(right + space_bit <= result_img.shape[1] - 1):
        right += space_bit
    else:
        right = result_img.shape[1] - 1

    new_img = result_img[up:down, left:right]

    # print(new_img.shape)

    # print('膨張する前の画像')
    # plt.imshow(new_img)
    # plt.show()

    dilation_img = new_img
    # dilate_var = 20
    # kernel = np.ones((dilate_var, dilate_var), np.uint8)
    # dilation_img = cv2.dilate(new_img, kernel, iterations = 1)

    # print('膨張した画像')
    # plt.imshow(dilation_img)
    # plt.show()

    h = int(dilation_img.shape[0] / IMG_ROWS) + 1
    w = int(dilation_img.shape[1] / IMG_COLS) + 1

    dilation_img = cv2.resize(dilation_img, (w * IMG_ROWS, h * IMG_COLS))

    resized_img = [[0] * IMG_COLS for i in range(IMG_ROWS)]

    max = 0
    total = 0
    cnt_not_zero = 0
    zero_or_not = [[0] * IMG_COLS for i in range(IMG_ROWS)]
    for i in range(IMG_ROWS):
        for j in range(IMG_COLS):
            cnt = 0
            for k in range(i * h, (i + 1) * h):
                for l in range(j * w, (j + 1) * w):
                    if(dilation_img[k][l] == 255):
                        cnt += 1
                        total += 1
            if(cnt != 0):
                cnt_not_zero += 1
                zero_or_not[i][j] += 1
            
            if(max < cnt):
                max = cnt
            resized_img[i][j] = cnt
        
    # print(max)

    resized_img = np.array(resized_img)

    resized_img = resized_img.astype('float32')

    # ave = total / cnt_not_zero

    # standard_diviation = std_divi(resized_img, ave)

    resized_img /= max
    resized_img *= 255
    for i in range(IMG_COLS):
        for j in range(IMG_ROWS):
            if(resized_img[i][j] >= 150):
                resized_img[i][j] = 255

    resized_rev_img = resized_img

    # plt.imshow(resized_rev_img)
    # plt.show()

    # file_name, ext = os.path.splitext( os.path.basename(file_path) )
    # new_path = './image_recognize/image_bit/' + file_name + '_bit' + ext

    # new_file = pathlib.Path(new_path)
    # new_file.touch()

    # cv2.imwrite(new_path, resized_rev_img)

    subfig.append(fig.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 3))
    subfig[2].imshow(resized_rev_img)
    plt.show()

    resized_rev_img = [resized_rev_img]

    resized_rev_img = np.array(resized_rev_img)

    if Keras.image_data_format() == 'channel_first':
        resized_rev_img = resized_rev_img.reshape(1, 1, IMG_ROWS, IMG_COLS)
    else:
        resized_rev_img = resized_rev_img.reshape(1, IMG_ROWS, IMG_COLS, 1)

    print('fin')
    
    return resized_rev_img