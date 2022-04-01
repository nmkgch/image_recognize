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

def extract_frame(img, file_path, average_hsv):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    bgrLower=np.array([0, 0, 0])
    bgrUpper=np.array([255, 90, 255])

    mask = cv2.inRange(hsvImg, bgrLower, bgrUpper)

    plt.imshow(mask)
    plt.show()

    mask_not = cv2.bitwise_not(mask)

    _img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = cv2.bitwise_and(_img, _img, mask=mask_not)

    chara_list = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(cv2.contourArea(contour))
    print("hierarchy:", hierarchy)

    for i in range(len(contours)):
        if(cv2.contourArea(contours[i]) <= 100):
            restore_hierarchy(hierarchy, i)
            continue
        if(cv2.contourArea(contours[i]) >= (img.shape[0] - 1) * (img.shape[1] - 1)):
            restore_hierarchy(hierarchy, i)
            continue

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

    area_top = img.shape[0] - 1
    area_bottom = 0
    area_left = img.shape[1] - 1
    area_right = 0
    for x in range(1, character_mask.shape[0] - 1):
        for y in range(1, character_mask.shape[1] - 1):
            if(character_mask[x,y] == 255):

                if(character_mask[x+1, y] == 0):
                    if(area_right < x):
                        area_right = x

                if(character_mask[x-1,y] == 0):
                    if(area_left > x):
                        area_left = x
                
                if(character_mask[x,y-1] == 0):
                    if(area_top > y):
                        area_top = y
                
                if(character_mask[x,y+1] == 0):
                    if(area_bottom < y):
                        area_bottom = y
            else: continue

    character_img = img[area_left:area_right, area_top:area_bottom]
    
    file_name, ext = os.path.splitext( os.path.basename(file_path) )
    new_path = './image_recognize/image_extract/' + file_name + '_extract' + ext

    new_file = pathlib.Path(new_path)
    new_file.touch()

    cv2.imwrite(new_path, character_img)

    return character_img