#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Reimplementation of the TensorFlow official CIFAR-10 CNN tutorials.

- 1. This model has 1,068,298 parameters, after few hours of training with GPU,
accuracy of 86% was found.

- 2. Note: The optimizers between official code and this code are different.

Links
-------
.. https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html
.. https://github.com/tensorflow/tensorflow/tree/r0.9/tensorflow/models/image/cifar10


Description
-----------
The images are processed as follows:
.. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
.. They are approximately whitened to make the model insensitive to dynamic range.

For training, we additionally apply a series of random distortions to
artificially increase the data set size:
.. Randomly flip the image from left to right.
.. Randomly distort the image brightness.
.. Randomly distort the image contrast.

Speed Up
--------
Reading images from disk and distorting them can use a non-trivial amount
of processing time. To prevent these operations from slowing down training,
we run them inside 32 separate threads which continuously fill a TensorFlow
queue.

"""

import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

"""
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('*** GPU device not found ***')
print('### Found GPU at: {} ###'.format(device_name))
"""

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
    shape=(-1, 32, 32, 3), plotable=False)


def print_dataset_shape(X_name, X, y_name, y):
    print(X_name + '.shape ', X.shape, end='\t')
    print(y_name + '.shape ', y.shape)
    print('%s.dtype %s , %s.dtype %s' % (X_name, X.dtype, y_name, y.dtype))


print_dataset_shape('X_train', X_train, 'y_train', y_train)
# (50000, 32, 32, 3)
# (50000,)
# X_train.dtype float32 , y_train.dtype int32

print_dataset_shape('X_test', X_test, 'y_test', y_test)
# (10000, 32, 32, 3)
# (10000,)
# X_test.dtype float32 , y_test.dtype int32


def data_to_tfrecord(images, labels, filename):
    """Save data into TFRecord."""
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return

    print("Current directory: %s " % os.getcwd())
    print("Converting data into %s ..." % filename)

    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        label = int(labels[index])
        img_raw = img.tobytes()

        # Visualize a image
        if index == 0:
            tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1,
                               saveable=False, name='label: ' + str(label), fig_idx=1236)

        # Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


# Save data into TFRecord files
data_to_tfrecord(images=X_train, labels=y_train, filename="train.cifar10")
data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar10")


def read_and_decode(filename, is_train=None):
    """Return tensor to read from TFRecord."""
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train == True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])

        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)

        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)

        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

        # 5. Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)

        # 2. Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)

    return img, label


def model(x_crop, y_, reuse):
    """For more simplified CNN APIs, check tensorlayer.org."""
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)

    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(x_crop, name='input')
        net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                     padding='SAME', W_init=W_init, name='cnn1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')
        net = LocalResponseNormLayer(
            net, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                     padding='SAME', W_init=W_init, name='cnn2')
        net = LocalResponseNormLayer(
            net, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool2')

        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, 384, act=tf.nn.relu,
                         W_init=W_init2, b_init=b_init2, name='d1relu')
        net = DenseLayer(net, 192, act=tf.nn.relu,
                         W_init=W_init2, b_init=b_init2, name='d2relu')
        net = DenseLayer(net, n_units=10, act=None,
                         W_init=W_init2, name='output')
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')

        """ 需给后面的全连接层引入L2 normalization，惩罚模型的复杂度，避免overfitting """
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # 加上L2模型复杂度惩罚项后，得到最终真正的cost
        cost = ce + L2

        # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def model_batch_norm(x_crop, y_, reuse, is_train):
    """Batch normalization should be placed before rectifier."""
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)

    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(x_crop, name='input')
        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME',
                     W_init=W_init, b_init=None, name='cnn1')
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')

        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME',
                     W_init=W_init, b_init=None, name='cnn2')
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch2')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool2')

        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, 384, act=tf.nn.relu,
                         W_init=W_init2, b_init=b_init2, name='d1relu')
        net = DenseLayer(net, 192, act=tf.nn.relu,
                         W_init=W_init2, b_init=b_init2, name='d2relu')
        net = DenseLayer(net, n_units=10, act=None,
                         W_init=W_init2, name='output')
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')

        """ 需给后面的全连接层引入L2 normalization，惩罚模型的复杂度，避免overfitting """
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # 加上L2模型复杂度惩罚项后，得到最终真正的cost
        cost = ce + L2

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


# Example to visualize data
# img, label = read_and_decode("train.cifar10", None)
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=4,
#                                                 capacity=50000,
#                                                 min_after_dequeue=10000,
#                                                 num_threads=1)
# print("img_batch   : %s" % img_batch._shape)
# print("label_batch : %s" % label_batch._shape)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     for i in range(3):  # number of mini-batch (step)
#         print("Step %d" % i)
#         val, l = sess.run([img_batch, label_batch])
#         # exit()
#         print(val.shape, l)
#         tl.visualize.images2d(val, second=1, saveable=False, name='batch'+str(i), dtype=np.uint8, fig_idx=2020121)
#
#     coord.request_stop()
#     coord.join(threads)
#     sess.close()

batch_size = 128
model_file_name = "./model_cifar10_advanced.ckpt"
resume = True  # load model, resume from previous checkpoint?

with tf.device('/cpu:0'):
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    """ Prepare data in cpu """
    # the tensors for reader and distorter for a single Example
    x_train_, y_train_ = read_and_decode("train.cifar10", True)
    x_test_, y_test_ = read_and_decode("test.cifar10", False)

    print_dataset_shape('x_train', x_train_, 'y_train', y_train_)
    # (24, 24, 3)
    # ()
    print(x_train_)
    print(y_train_)

    print_dataset_shape('x_test', x_test_, 'y_test', y_test_)
    print(x_test_)
    print(y_test_)

    # the tensors for streaming to the model with a batch augmented data per a training batch/step
    # by using multi-threads.
    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        [x_train_, y_train_], batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=32
        # set the number of threads here
    )

    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch(
        [x_test_, y_test_], batch_size=batch_size, capacity=50000, num_threads=32
    )

    print_dataset_shape('x_train_batch', x_train_batch, 'y_train_batch', y_train_batch)
    # (batch_size, 24, 24, 3)
    # (batch_size,)
    print(x_train_batch)
    print(y_train_batch)

    print_dataset_shape('x_test_batch', x_test_batch, 'y_test_batch', y_test_batch)
    print(x_test_batch)
    print(y_test_batch)

    # You can also use placeholder to feed_dict in data after using
    # val, l = sess.run([x_train_batch, y_train_batch]) to pull a batch size of data
    #
    # Demo:
    #   # Define the model using placeholder
    #   x_crop = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    #   y_ = tf.placeholder(tf.int32, shape=[batch_size,])
    #   cost, acc, network = model(x_crop, y_, None)
    #
    #   ...
    #
    #   # In the loop of batches
    #   for b in range(batch count in a epoch):
    #     val, l = sess.run([x_train_batch, y_train_batch])   # pull a batch size of data
    #
    #     tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
    #     err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        # using local response normalization
        network, cost, acc, = model(x_train_batch, y_train_batch, False)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, True)

        # you may want to try batch normalization
        # network, cost, acc, = model_batch_norm(x_train_batch, y_train_batch, None, is_train=True)
        # _, cost_test, acc_test = model_batch_norm(x_test_batch, y_test_batch, True, is_train=False)

    # train
    n_epoch = 50000
    learning_rate = 0.0001
    print_freq = 1
    n_step_epoch = int(len(y_train) / batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tl.layers.initialize_global_variables(sess)
    if resume and os.path.isfile(model_file_name):
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' %
          (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):
            # You can also use placeholder to feed_dict in data after using
            # val, l = sess.run([x_train_batch, y_train_batch]) to pull a batch size of data
            #
            # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
            # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
            err, ac, _ = sess.run([cost, acc, train_op])
            step += 1
            train_loss += err
            train_acc += ac
            n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print(
                "Epoch %d : Step %d-%d of %d took %fs" %
                (epoch, step - n_step_epoch, step,
                 n_step, time.time() - start_time)
            )
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(len(y_test) / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

        if (epoch + 1) % (print_freq * 50) == 0:
            print("Save model " + "!" * 10)
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)
            # you can also save model into npz
            tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
            # and restore it as follow:
            # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    coord.request_stop()
    coord.join(threads)
    sess.close()
