---
typora-root-url: ./
---

# License-plate-RC

There are two items in the directory 。

1. Car-RC

2. Plate Recognition

   ​

The first project identifies preprocessed license plate characters. 

![](/Car-RC/car/0-000e-4064-85c.jpg)

Modeling with convolutional neural networks

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


In the second project, the license plates taken on the spot were directly recognized.

![](/PlateRecognition/output.png)

Based on HyperLPR Source Framework（hyperlpr High Performance Open Source Chinese License Plate Recognition Framework ）

python -m pip install hyperlpr





