#! /usr/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
import tensorlayer as tl

# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()
# Use tfdbg CLI
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# Use tfdbg dashboard
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "albert-mbp.local:6064")

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10, act=None, name='output')

# define cost function and metric.
""" y is a matrix with shape(batch_size, 10) """
y = network.outputs

""" cost is a matrix with shape(batch_size, 1) """
cost = tl.cost.cross_entropy(y, y_, name='cost')

""" tf.argmax(y, 1) is a vector of index of the maximum in per lines. i.e. [index_0, index_1, ..., index_batch_size] """
""" y_ is a vector [y0_label, y1_label, ...], and the length of y_ is the batch_size. """
""" tf.equal() returns a vector [TrueOrFalse_0, TrueOrFalse_1, ..., TrueOrFalse_batch_size] """
correct_prediction = tf.equal(tf.argmax(y, 1), y_)

""" tf.cast() returns [0 or 1, 0 or 1, ...] """
""" caculating the mean value of correct_predictions of a batch. i.e. sum(correct_predictions of a batch)/batch_size """
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
# Comment below line it is deprecated in favor of tf.global_variables_initializer()
#tl.layers.initialize_global_variables(sess)

"""
If tensorboard=True, the `global_variables_initializer` will be run inside the fit() function
in order to initialize the automatically generated summary nodes used for tensorboard visualization,
thus `tf.global_variables_initializer().run()` before the `fit()` call will be undefined.
"""
# Comment below line because I will using tensorboard=True when invoking tf.utils.fit()
#sess.run(tf.global_variables_initializer())


# print network information
"""[TL] Attempting to use uninitialized value relu1/W"""
network.print_params(False)
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_, acc=acc, batch_size=500, \
  n_epoch=500, print_freq=5, X_val=X_val, y_val=y_val, eval_train=False, tensorboard=True)

# print network parameters after training.
network.print_params()

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# save the network to .npz file
tl.files.save_npz(network.all_params, name='model.npz')
sess.close()
