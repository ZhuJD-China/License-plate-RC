# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : create_dir.py
# @Time     : 2020/5/27 20:13
# @Software : PyCharm


import os
import shutil


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def copyPIC():
    # 复制图像到另一个文件夹
    # 文件所在文件夹
    file_dir = './'
    # 创建一个子文件存放文件
    name = 'class'

    file_list = os.listdir(file_dir)

    for image in file_list:

        # 如果图像名为B.png 则将B.png复制到F:\\Test\\TestA\\class
        if image == "B.png":
            if os.path.exists(os.path.join(file_dir, 'class_name')):
                shutil.copy(os.path.join(file_dir, image), os.path.join(file_dir, 'class_name'))
            else:
                os.makedirs(os.path.join(file_dir, 'class_name'))
                shutil.copy(os.path.join(file_dir, image), os.path.join(file_dir, 'class_name'))


if __name__ == '__main__':
    for index in range(0, 37):
        file = "./" + str(index)
        mkdir(file)  # 调用函数
