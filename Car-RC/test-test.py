# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : test-test.py
# @Time     : 2020/5/26 19:36
# @Software : PyCharm

import sys
import os
import time
import random
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from PIL import Image
import pandas as pd

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 38
iterations = 30
NUM_CHINESE_CHARACTER = 0
NUM_LETTERS = 26

# 存放模型和权重参数的文件夹
SAVER_DIR = "train-saver_me/digits/"

LETTERS_DIGITS = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "T", "U", "V", "X", "Y", "Z", "赣", "冀", "晋", "蒙", "鲁", "陕", "宁")
license_num = ""

time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


def carnum_test():
    dic = dict()
    license_num = ""
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (SAVER_DIR))
    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)

        # 第一个卷积层
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")  # sess.graph.get_tensor_by_name获取模型训练过程中的变量和参数名
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout层
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")

        # 定义优化器和训练op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # for n in range(2,8):
        '''
        path = "./data/"
        files = os.listdir(path)
        print(type(files[0]))
        print(len(files))
        for i in files:
            index = 0
            detial = os.listdir("./data/"+i+"/")
            for pic in detial:
        '''

        path = "./data/"
        files = os.listdir(path)
        for i in files:
            detial = os.listdir("./data/" + i + "/")

            for n in range(1, 8):
                # path = "img_test/%s.bmp" % (n)
                # path = "test_images/%s.bmp" % (n)
                path = "./data/" + i + "/%s.jpg" % (n)
                # path = "img/%s.bmp" % (n)
                try:
                    img = Image.open(path)
                except:
                    continue
                # img = img.resize((22,22), Image.BICUBIC)
                # img = cv2.imread(path)
                # img = cv2.resize(img, (32,40), interpolation=cv2.INTER_CUBIC)
                # info = img.shape  # 获取图片的宽 高 颜色通道信息
                # width = info[0]
                # height = info[1]
                width = img.size[0]
                height = img.size[1]
                #print("img_shape", img.size)
                img_data = [[0] * SIZE for i in range(1)]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) < 190:
                            img_data[0][w + h * width] = 1
                        else:
                            img_data[0][w + h * width] = 0

                result = sess.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})

                max1 = 0
                max2 = 0
                max3 = 0
                max1_index = 0
                max2_index = 0
                max3_index = 0
                for j in range(NUM_CLASSES + NUM_CHINESE_CHARACTER):
                    if result[0][j] > max1:
                        max1 = result[0][j]
                        max1_index = j
                        continue
                    if (result[0][j] > max2) and (result[0][j] <= max1):
                        max2 = result[0][j]
                        max2_index = j
                        continue
                    if (result[0][j] > max3) and (result[0][j] <= max2):
                        max3 = result[0][j]
                        max3_index = j
                        continue

                license_num = license_num + LETTERS_DIGITS[max1_index]
                # print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
                #     LETTERS_DIGITS[max1_index], max1 * 100, LETTERS_DIGITS[max2_index], max2 * 100,
                #     LETTERS_DIGITS[max3_index],
                #     max3 * 100))

            print("车牌编号是: 【%s】" % license_num)
            dic[str(i)] = license_num
            license_num = ""

    res = pd.DataFrame(pd.Series(dic), columns=['Recognition'])
    res.to_csv('./res.csv')

if __name__ == "__main__":
    carnum_test()
