'''
left：与左边界的距离

up：与上边界的距离

right：还是与左边界的距离

below：还是与上边界的距离
'''
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

path = "./data1/"
files = os.listdir(path)
for i in files[0:-1]:
    details = os.listdir("./data1/" + i + "/")

    for origin in details[1:-1]:
        try:
            img = Image.open("./data1/" + i + "/" + origin[0:1] + ".png")
            # img = Image.open("./data1/" + i + "/" + origin[0:1] + ".png")
        finally:
            print(img.size)

            # cropped = img.crop((0, 0, 50, 41))  # (left, upper, right, lower)
            # size = img.size
            # print('origin: ', size)
            # plt.figure("origin")

            size_cropped = img.size
            # print('cropped: ', size_cropped)
            #
            # # 如果图片width大于其mean 那么就尝试从中间纵向切割
            # print(size_cropped[0])
            # print(size_cropped[1])

            if size_cropped[0] >= 48:
                cropped_new_left = img.crop((0, 0, size_cropped[0] // 2, 41))
                cropped_new_right = img.crop((size_cropped[0] // 2, 0, size_cropped[0], size_cropped[1]))
                # plt.imshow(cropped)
                plt.imshow(cropped_new_left)
                filename = origin.split(".", 1)
                cropped_new_left.save("./data1/" + i + "/" + filename[0] + "_1.png")
                plt.imshow(cropped_new_right)
                cropped_new_right.save("./data1/" + i + "/" + filename[0] + "_2.png")
                os.remove("./data1/" + i + "/" + filename[0] + ".png")
                plt.show()
            else:
                print("<48")

                # cropped.save("./pil_cut_thor.jpg")


# 将所有图片resize 32*40
for i in files[0:-1]:
    details = os.listdir("./data1/" + i + "/")
    for origin in details[0:-1]:
        filename = origin.split(".", 1)
        img = cv2.imdecode(np.fromfile("./data1/" + i + "/" + filename[0] + ".png", dtype=np.uint8), cv2.IMREAD_COLOR)
        m = 32 * img.shape[0] / img.shape[1]
        # 压缩图像
        img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)
        print(str(32 * int(m)))
        cv2.imwrite("./data1/" + i + "/" + filename[0] + ".jpg", img)

