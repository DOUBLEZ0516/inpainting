"""
author: carpedm20@github
editor: Wentian Bao, wb2328
        Zhang Zhang, zz2517
        Hangtian Zhang, hz2475
"""
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

#be compatible with different tensorflow versions
#Wentian Bao wb2328, Zhang Zhang zz2517
try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def batch_norm(x, train=True, epsilon=1e-5, momentum = 0.9):
  return tf.contrib.layers.batch_norm(x,
                    decay=self.momentum, 
                    updates_collections=None,
                    epsilon=self.epsilon,
                    scale=True,
                    is_training=train)

def conv_cond_concat(x, y):
  """
  Wentian Bao wb2328
  Concatenate conditioning vector on feature map axis.
  """
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  """
  Wentian Bao wb2328
  Zhang Zhang zz2517
  convolution operatrion for GAN
  """
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  """
  Wentian Bao wb2328
  Zhang Zhang zz2517
  deconvolution operation for GAN
  """
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
              strides=[1, d_h, d_w, 1])

    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  """
  Wentian Bao wb2328
  Zhang Zhang zz2517
  leaky relu function
  """
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  """
  Wentian Bao wb2328
  Zhang Zhang zz2517
  linear transformation
  """
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
