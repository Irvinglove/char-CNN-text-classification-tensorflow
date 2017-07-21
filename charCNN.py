# coding=utf-8
import tensorflow as tf
from data_helper import Dataset
from math import sqrt
from config import config


class CharCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, l0, num_classes, conv_layers, fc_layers, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, l0], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            train_data = Dataset(config.train_data_source)
            self.W, _ = train_data.onehot_dic_build()
            self.x_image = tf.nn.embedding_lookup(self.W, self.input_x)
            self.x_flat = tf.expand_dims(self.x_image, -1)

        for i, cl in enumerate(conv_layers):
            with tf.name_scope("conv_layer-%s" % (i+1)):
                print "开始第" + str(i + 1) + "卷积层的处理"
                filter_width = self.x_flat.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])
                w_conv = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),
                                     dtype='float32', name='w')
                # w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
                b_conv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')
                # b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                conv = tf.nn.conv2d(
                    self.x_flat,
                    w_conv,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                h_conv = tf.nn.bias_add(conv, b_conv)

                if not cl[-1] is None:
                    ksize_shape = [1, cl[2], 1, 1]
                    h_pool = tf.nn.max_pool(
                        h_conv,
                        ksize=ksize_shape,
                        strides=ksize_shape,
                        padding='VALID',
                        name='pool')
                else:
                    h_pool = h_conv

                self.x_flat = tf.transpose(h_pool, [0, 1, 3, 2], name='transpose')

        with tf.name_scope('reshape'):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value
            self.x_flat = tf.reshape(self.x_flat, [-1, fc_dim])

        weights = [fc_dim] + fc_layers
        for i, fl in enumerate(fc_layers):
            with tf.name_scope('fc_layer-%s' % (i+1)):
                print "开始第" + str(i + 1) + "全连接层的处理"
                stdv = 1 / sqrt(weights[i])
                w_fc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv),
                                   dtype='float32', name='w')
                b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name='b')
                # 不同的初始化方式
                # w_fc = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                # b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")
                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, w_fc) + b_fc)

                with tf.name_scope('drop_out'):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_keep_prob)

        with tf.name_scope('output_layer'):
            print "开始输出层的处理"
            # w_out = tf.Variable(tf.truncated_normal([fc_layers[-1], num_classes], stddev=0.1), name="W")
            # b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            stdv = 1 / sqrt(weights[-1])
            w_out = tf.Variable(tf.random_uniform([fc_layers[-1], num_classes], minval=-stdv, maxval=stdv),
                                dtype='float32', name='W')
            b_out = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stdv, maxval=stdv), name='b')
            self.y_pred = tf.nn.xw_plus_b(self.x_flat, w_out, b_out, name="y_pred")
            self.predictions = tf.argmax(self.y_pred, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
