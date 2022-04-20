from module_ImageRecognize import extract_frame
import matplotlib.pyplot as plt
import cv2

# 画像はBGR形式
img = cv2.imread("./image_recognize/frame_point_image/WIN_20220420_21_19_53_Pro.jpg")

print(img.shape)

plt.imshow(img)
plt.show()

character_img_array, frame_number = extract_frame.extract_frame(img, 0)

print(len(character_img_array), frame_number)