# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : ZhuJD
# @FILE     : train-test.py
# @Time     : 2020/5/26 19:36
# @Software : PyCharm

import sys
import os
import time
import random

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
import cv2


SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 38
iterations = 30
NUM_CHINESE_CHARACTER = 0
NUM_LETTERS = 26

#存放模型和权重参数的文件夹
SAVER_DIR = "train-saver_me/digits/"
LETTERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "T", "U", "V", "X", "Y", "Z", "赣", "冀", "晋", "蒙", "鲁", "陕", "宁")
license_num = ""

time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES+NUM_CHINESE_CHARACTER])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

train_dirlist = []
validate_dirlist = []
def gen_dir():
    for i in range(0, NUM_CLASSES):
        dir = './train_images/mytraining-set/%s/' % i  # 这里可以改成你自己的图片文件夹
        train_dirlist.append(dir)

    # for i in range(0, NUM_CHINESE_CHARACTER):
    #     dir = './train_images/training-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片文件夹
    #     train_dirlist.append(dir)

    # for i in range(0, NUM_LETTERS):
    #     dir = './train_images/training-set/letters/%s/' % (i+10)  # 这里可以改成你自己的图片文件夹
    #     dirlist.append(dir)

    # for dir in dirlist:
    #     print(dir)

    for i in range(0, NUM_CLASSES):
        dir = './train_images/myvalidation-set/%s/' % i  # 这里可以改成你自己的图片文件夹
        validate_dirlist.append(dir)

    # for i in range(0, NUM_CHINESE_CHARACTER):
    #     dir = './train_images/validation-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片文件夹
    #     validate_dirlist.append(dir)

    return

# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

# 定义训练网络
def carnum_train():
    iterations = 30
    time_begin = time.time()
    # 第一次遍历图片文件夹是为了获取图片总数
    input_count = 0
    # for i in range(0, NUM_CLASSES):
        # dir = './train_images/training-set/%s/' % i  # 这里可以改成你自己的图片文件夹
    for dir in train_dirlist:
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1

    # 定义对应维数和各维长度的数组
    input_images = np.array([[0] * SIZE for i in range(input_count)])
    input_labels = np.array([[0] * (NUM_CLASSES+NUM_CHINESE_CHARACTER) for i in range(input_count)])

    # 第二次遍历图片文件夹是为了生成图片数据和标签，里面都是一堆0和1，因为都是二值化图像
    index = 0
    i = 0
    # for i in range(0, NUM_CLASSES):
        # dir = './train_images/training-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for dir in train_dirlist:
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                # print(filename+":"+str(width)+","+str(height))
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:  # img.getpixel遍历一张图像的所有像素点
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                        # print(str(index)+", "+str(w+h*width))
                input_labels[index][i] = 1
                index += 1
        i += 1


    # 第一次遍历图片目录是为了获取图片总数
    val_count = 0
    # for i in range(0, NUM_CLASSES):
    #     dir = './train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for dir in validate_dirlist:
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1

    # 定义对应维数和各维长度的数组
    val_images = np.array([[0] * SIZE for i in range(val_count)])
    val_labels = np.array([[0] * (NUM_CLASSES+NUM_CHINESE_CHARACTER) for i in range(val_count)])

    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    i = 0
    # for i in range(0, NUM_CLASSES):
    #     dir = './train_images/validation-set/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for dir in validate_dirlist:
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w + h * width] = 0
                        else:
                            val_images[index][w + h * width] = 1
                val_labels[index][i] = 1
                index += 1
        i += 1

    with tf.Session() as sess:
        # 第一个卷积层
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES+NUM_CHINESE_CHARACTER], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES+NUM_CHINESE_CHARACTER]), name="b_fc2")

        # 定义优化器和训练op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # 求交叉熵
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # Adam优化
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 运行整个网络
        sess.run(tf.global_variables_initializer())

        time_elapsed = time.time() - time_begin
        print("读取图片文件耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        print("一共读取了 %s 个训练图像， %s 个标签" % (input_count, input_count))

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，
        # 譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

        # 执行训练迭代
        for it in range(iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):  # n是从0到72
                train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                          y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size
                train_step.run(feed_dict={x: input_images[start_index:input_count - 1],
                                          y_: input_labels[start_index:input_count - 1], keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it % 5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
                if iterate_accuracy >= 0.9999 and it >= iterations:
                    break

        print('完成训练!')
        time_elapsed = time.time() - time_begin
        print("训练耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(SAVER_DIR)
        # 初始化saver
        saver = tf.train.Saver()  # 这是tensorflow提供的API接口，用来保存和还原一个神经网络
        saver_path = saver.save(sess, "%smodel.ckpt" % (SAVER_DIR))

def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    print(str(w)+", "+str(h))

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocess():
    gen_dir()
    for dir in train_dirlist:
        i = 1
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),cv2.IMREAD_COLOR)
                m = 32 * img.shape[0] / img.shape[1]

                # 压缩图像
                img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)

                # BGR转换为灰度图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dir+"%s.png"%i, gray_img)
                i += 1

    for dir in validate_dirlist:
        i = 1
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),cv2.IMREAD_COLOR)
                m = 32 * img.shape[0] / img.shape[1]

                # 压缩图像
                img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)

                # BGR转换为灰度图像
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dir+"%s.png" % i, gray_img)
                i += 1

if __name__=="__main__":
    gen_dir()
    carnum_train()

