from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.python.framework import graph_util

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

#define the DCGAN model
class DCGAN(object):
  def __init__(self, sess, input_height=128, input_width=128, crop=False,
         batch_size=64, sample_num = 64, output_height=128, output_width=128,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=4096, dfc_dim=4096, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
     Return:
      a DCGAN instance
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    imreadImg = imread(self.data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    """
    Use initial values to build networks
    """
    #Whether images need to be cropped
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]
    
    #dim(inputs) = batch_size * input_height * input_wideth * channel
    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs
    
    #dim(z) = batch_size * z_dim
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z) # get images generated from generator
    self.D_real, self.D_real_logits   = self.discriminator(inputs, reuse=False) # get real images' probabilities and logits from discriminator
    self.D_gen, self.D_gen_ogits = self.discriminator(self.G, reuse=True) # get generated images' probabilities and logits from discriminator
    
    self.d_sum = histogram_summary("d", self.D_real)
    self.d_sum_gen = histogram_summary("d_", self.D_gen)
    self.G_sum = image_summary("G", self.G)
    
    #calculate loss of discriminator and generator using logits from discriminator
    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen_logits, labels=tf.zeros_like(self.D_gen)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_gen_logits, labels=tf.ones_like(self.D_gen)))
    self.d_loss = self.d_loss_real + self.d_loss_fake

    #keep track of losses of generator and discriminator in tensorboard
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    all_vars = tf.trainable_variables()
    #separate variables from discriminator and generator for training
    self.d_vars = [var for var in all_vars if 'd_' in var.name]
    self.g_vars = [var for var in all_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """
    main worker function for training.
    """
    #d_optimizor only optimizes variables of discriminator
    #g_optimizor only optimizes variables of generator
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    
    tf.global_variables_initializer().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)
    
    #sample latent variables from latent space in uniform distribution
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    #read images from files
    sample_files = self.data[0:self.sample_num]
    sample = [
        get_image(sample_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for sample_file in sample_files]
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)
    #counter counts the iteration number 
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    sess = tf.Session()
    #our default is 25 epoches
    for epoch in xrange(config.epoch):     
      self.data = glob(os.path.join(
        "./data", config.dataset, self.input_fname_pattern))
      batch_idxs = min(len(self.data), config.train_size) // config.batch_size
      #for the batch number in one epoch
      for idx in xrange(0, batch_idxs):
        batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        
        d_steps = 1
        g_steps = 2
        
        # Update D network for d-steps
        # input is a batch of real iamges and samples of z from latent space
        for i in range(d_steps):
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
        
        for i in range(g_steps):
          # Update G network for g-steps
          # input is a batch of samples from latent space
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
         
        #Get loss of discriminator and generator
        errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        errG = self.g_loss.eval({self.z: batch_z})
        
        #One iteration finished, add counter by 1
        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, config.epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))
        
        #get samples from generator every 100 iterations
        if np.mod(counter, 100) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.generator, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          except:
            print("one pic error!...")
        
        # For every 500 iterations save model in checkpoints
        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)
        
        #For every 1000 iterations save the model as a .pb file, which is easier to load than checkpoints.
        if np.mod(counter, 1000) == 1:
          op_list = ['Mean_2', 'generator/Tanh', 'discriminator/Sigmoid']
          graph_def = tf.get_default_graph().as_graph_def()
          constant_graph = graph_util.convert_variables_to_constants(self.sess, graph_def, op_list)
          with tf.gfile.FastGFile('./../semantic_image_inpainting/graphs/model_128_64_64_1.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
          #tf.train.write_graph(constant_graph, './checkpoint', 'model4.pb', as_text=False)

  #Define Discriminator
  def discriminator(self, image, reuse=False):
  """
  Arguments:
  image: images from training set and generator;
  reuse: whether reuse the discriminator;
  Return:
  probabilities and logits of the images in a batch
  """
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      
      #the network of discriminator, which contains 4 deconvolution layers. 
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4
  
  #define generator
  def generator(self, z):
    """
    Arguments:
    z: default 100 dimensional vectors from latent spaces, sampled from uniform distribution;
    Return:
    tf.nn.tanh(h4), dim = batch_size * output_height * output_width * channel
    """
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      #Network of generator, which contains 4 convolution layers.
      h0 = tf.nn.relu(batch_norm( tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])))
      h1 = tf.nn.relu(batch_norm(deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1'e)))
      h2 = tf.nn.relu(batch_norm(deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')))
      h3 = tf.nn.relu(batch_norm(deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')))
      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
      return tf.nn.tanh(h4)

  @property
  def model_dir(self):
  """
  Return the model's name 
  """
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
   """
   Save the model in checkpoints
   """
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
  """
  Load checkpoints
  """
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
