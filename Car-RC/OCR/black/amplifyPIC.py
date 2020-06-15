# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : amplifyPIC.py
# @Time     : 2020/5/26 23:58
# @Software : PyCharm

import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

path = "./"
files = os.listdir(path)
print(type(files[0]))
print(len(files))
index = 0
for pic in files:
    # files = os.listdir("./"+i+"/")
    index = index + 1
    print(pic)
    img = cv2.imdecode(np.fromfile("./" + pic, dtype=np.uint8), cv2.IMREAD_COLOR)
    m = 32 * img.shape[0] / img.shape[1]
    # 压缩图像
    img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)
    print(str(32 * int(m)))
    # BGR转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./amplify_result/pic" + str(index) + ".png", gray_img)

    img = Image.open("./amplify_result/pic" + str(index) + ".png")
    width = img.size[0]
    height = img.size[1]
    print(str(width) + "," + str(height))
    print('gray_img.shape', gray_img.shape)
    # fig = plt.figure(figsize=(10, 15))
    # # fig.add_subplot(1, 1, 1)
    # # plt.title("raw image")
    # # plt.imshow(gray_img)
    # # plt.show()

