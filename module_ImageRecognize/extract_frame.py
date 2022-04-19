import cv2
import numpy as np
from PIL import Image
import os
import pathlib
import matplotlib.pyplot as plt

def restore_hierarchy(hierarchy, index):
    if(hierarchy[0][index][0] != -1):
        hierarchy[0][hierarchy[0][index][0]][1] = hierarchy[0][index][1]
    if(hierarchy[0][index][1] != -1):
        hierarchy[0][hierarchy[0][index][1]][0] = hierarchy[0][index][0]
    if(hierarchy[0][index][2] != -1):
        i = hierarchy[0][index][2]
        while(i != -1):
            hierarchy[0][i][3] = -1
            i = hierarchy[0][i][0]
        i = hierarchy[0][index][2]
        while(i != -1):
            hierarchy[0][i][3] = -1
            i = hierarchy[0][i][1]
    if(hierarchy[0][index][3] != -1):
        hierarchy[0][hierarchy[0][index][3]][2] = hierarchy[0][index][0]
        if(hierarchy[0][hierarchy[0][index][3]][2] == -1):
            hierarchy[0][hierarchy[0][index][3]][2] = hierarchy[0][index][1]
        else:
            i = hierarchy[0][hierarchy[0][index][0]][0]
            while(i != -1):
                hierarchy[0][hierarchy[0][index][3]][2] = i
                i = hierarchy[0][i][0]


    hierarchy[0][index][0] = -1
    hierarchy[0][index][1] = -1
    hierarchy[0][index][2] = -1
    hierarchy[0][index][3] = -1
    return

def erase_exception(img, contours, hierarchy):
    # contours_io:どの輪郭が残っているかを記憶
    contours_io = np.ones(len(contours))
    for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) <= 100):
            restore_hierarchy(hierarchy, i)
            contours_io[i] = 0
            continue
        if(cv2.contourArea(contours[i]) >= (img.shape[0] - 1) * (img.shape[1] - 1)):
            restore_hierarchy(hierarchy, i)
            contours_io = 0
            continue
    return contours_io

def find_area(img):
    area_top = img.shape[0] - 1
    area_bottom = 0
    area_left = img.shape[1] - 1
    area_right = 0
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            if(img[x,y] == 255):

                if(img[x-1,y] == 0):
                    if(area_top > x):
                        area_top = x
                
                if(img[x+1, y] == 0):
                    if(area_bottom < x):
                        area_bottom = x

                if(img[x,y-1] == 0):
                    if(area_left > y):
                        area_left = y
                
                if(img[x,y+1] == 0):
                    if(area_right < y):
                        area_right = y
            else: continue
    return area_top, area_bottom, area_left, area_right

def extract_point(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hsvLower = np.array([150, 50, 0])
    hsvUpper = np.array([200, 255, 255])

    mask = cv2.inRange(hsvImg, hsvLower, hsvUpper)

    plt.imshow(mask)
    plt.show()

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(contours)

    print("処理する前:", hierarchy)
    
    contours_io = erase_exception(img, contours, hierarchy)

    print("処理したあと:", hierarchy)

    point_locate = [None for i in range(len(contours))]
    for i in range(len(contours)):
        if(contours_io[i] != 1):
            continue
        point_img = np.zeros((img.shape[0], img.shape[1]))
        cv2.fillConvexPoly(point_img, contours[i], 255)
        area_top, area_bottom, area_left, area_right = find_area(point_img)

        mid_point = [int((area_top + area_bottom) / 2), int((area_left + area_right) / 2)]

        point_locate[i] = mid_point

    return point_locate

def extract_frame(img, file_path):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hsvLower=np.array([215, 50, 0])
    hsvUpper=np.array([255, 255, 255])

    mask = cv2.inRange(hsvImg, hsvLower, hsvUpper)

    plt.imshow(mask)
    plt.show()

    _img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = cv2.bitwise_and(_img, _img, mask=mask)

    chara_list = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # [Next, Previous, First_Child, Parent]

    for contour in contours:
        print(cv2.contourArea(contour))
    print("hierarchy:", hierarchy)

    erase_exception(img, contours, hierarchy)

    print("hierarchy:",hierarchy)
    
    parent = None
    for i in range(len(contours)):
        if(hierarchy[0][i][2] != -1):
            parent = i
            break

    print("parent:", parent)

    extract_index = None
    while(parent != -1):
        extract_index = parent
        parent = hierarchy[0][parent][2]

    print("extract_index", extract_index)

    empty_img = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(empty_img, contours[extract_index], 255)
    character_mask = empty_img

    area_top, area_bottom, area_left, area_right = find_area(img)

    character_img = img[area_top:area_bottom + 1, area_left:area_right + 1]
    
    file_name, ext = os.path.splitext( os.path.basename(file_path) )
    new_path = './image_recognize/image_extract/' + file_name + '_extract' + ext

    new_file = pathlib.Path(new_path)
    new_file.touch()

    cv2.imwrite(new_path, character_img)

    return character_img