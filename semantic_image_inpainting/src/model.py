import tensorflow as tf
import numpy as np
import external.poissonblending as blending
from scipy.signal import convolve2d

class ModelInpaint():
    """
     This function defines the model of Semantic Inpainting
     including:
     1. load GAN
     2. mask input images
     3. forwardpropagation and backpropagation over the model
     4. genertate output images
     """
    # zz2517 wb2328
    def __init__(self, modelfilename, config,
                 model_name='dcgan',
                 gen_input='z:0',
                 gen_output='Tanh:0',
                 gen_loss='Mean_2:0',
                 disc_input='real_images:0',
                 disc_output='Sigmoid:0',
                 z_dim=100,
                 batch_size=64):

        """
        Initialization function, takes user defined arguments to define the model structure

        Parameters:
        modelfilename: tensorflow .pb file with weights to be loaded
        config: training parameters: lambda_p, nIter
        gen_input: node name for generator input
        gen_output: node name for generator output
        disc_input: node name for discriminator input
        disc_output: node name for discriminator output
        z_dim - latent space dimension of GAN
        batch_size - training batch size
        """
        self.config = config

        self.batch_size = batch_size
        self.z_dim = z_dim
        self.graph, self.graph_def = ModelInpaint.loadpb(modelfilename,
                                                         model_name)
        self.gi, self.go, self.gl, self.di, self.do = ModelInpaint.load_layers(self.graph, modelfilename,
                                                                               model_name, gen_input,
                                                                               gen_output, gen_loss,
                                                                               disc_input, disc_output)

        self.image_shape = self.go.shape[1:].as_list()
        self.l = config.lambda_p
        self.sess = tf.Session(graph=self.graph)
        self.z = np.random.randn(self.batch_size, self.z_dim)
        self.Wstep = config.Wstep


    # zz2517 wb2328
    def prepare(self, images, imask, nsize=7):
        """
        This function performs
        1. image preprocessing: prepare input image to feed the model
        2. mask manipulation: use input mask to generate weighted mask
                              and duplicate the weighted mask from 2D to 3D

        images: input images
        imask: input mask
        nsize: window size to generate weighted mask
        """
        images = ModelInpaint.imtransform(images)
        self.masks_data = ModelInpaint.genWeightedMask(imask, self.batch_size, nsize)
        self.bin_mask = ModelInpaint.genBinarizeMask(imask, dtype='uint8')

        num_images = images.shape[0]
        self.images_data = np.repeat(images[np.newaxis, 0, :, :, :],
                                     self.batch_size,
                                     axis=0)
        ncpy = min(num_images, self.batch_size)
        self.images_data[:ncpy, :, :, :] = images[:ncpy, :, :, :].copy()

    # wb2328
    def blending(self, g_out, blend = True):
        """
        This function applies poisson blending using binary mask.

        g_out: generator output
        blend: determines what blending to use. True (default) for poisson
                blending and False to alpha blending
        """
        images_out = ModelInpaint.iminvtransform(g_out)
        images_in = ModelInpaint.iminvtransform(self.images_data)

        if blend:
            for i in range(len(g_out)):
                images_out[i] = ModelInpaint.poissonblending(
                    images_in[i], images_out[i], self.bin_mask
                )
        else:
            images_out = np.multiply(images_out, 1-self.masks_data) \
                         + np.multiply(images_in, self.masks_data)

        for i in range(len(g_out)):
            images_out[i,:,:,:] += images_in[i,0,0,0] - images_out[i,0,0,0]
        return images_out
    
    # wb2328
    def build_semantic_model(self):
        """
        This function takes into latent variable z, mask and image to calculate
        context loss, prior loss, and gradient on z and weight mask W
        """
        with self.graph.as_default():
            self.masks = tf.placeholder(tf.float32,
                                        [None] + self.image_shape,
                                        name='mask')
            self.images = tf.placeholder(tf.float32,
                                         [None] + self.image_shape,
                                         name='images')
            self.context_loss = tf.reduce_sum(
                    tf.contrib.layers.flatten(
                        tf.abs(tf.multiply(self.masks, self.go) -
                               tf.multiply(self.masks, self.images))), 1)

            self.perceptual_loss = self.gl
            self.inpaint_loss = self.context_loss + self.l*self.perceptual_loss
            self.inpaint_grad = tf.gradients(self.inpaint_loss, self.gi)
            #masks gradient
            self.masks_grad = tf.gradients(self.context_loss, self.masks)
    
    # zz2517
    def backpropagation(self, verbose=True):
        """
        This function performs backpropagation to input by gradient decent to
        obtain latent space representation of generated image, and returns
        generated output image
        """
        v = 0
        for i in range(self.config.nIter):
            out_vars = [self.inpaint_loss, self.inpaint_grad, self.go, self.masks_grad]

            in_dict = {self.masks: self.masks_data,
                       self.gi: self.z,
                       self.images: self.images_data}

            loss, grad, imout, masks_grad = self.sess.run(out_vars, feed_dict=in_dict)

            v_prev = np.copy(v)
            v = self.config.momentum*v - self.config.lr*grad[0]
            self.z += (-self.config.momentum * v_prev +
                       (1 + self.config.momentum) * v)
            self.z = np.clip(self.z, -1, 1)

            #update mask_data for every 100 iters
            if i % self.Wstep == 0:
                self.masks_data += self.config.lr * masks_grad[0]
                print(i)

            if verbose:
                print('Iteration {}: {}'.format(i, np.mean(loss)))

        return imout
    # wb2328
    def inpaint(self, image, mask, blend=True):
        """
        This function performs inpainting on image by calling

        image: input RGB image
        mask: input binary mask
        blend: boolean value to determine whether to perform poisson blending
        Return: inpainted images
        """
        self.build_semantic_model()
        self.prepare(image, mask)
        imout = self.backpropagation()
        return self.blending(imout, blend), imout

    # wb2328
    @staticmethod
    def loadpb(filename, model_name='dcgan'):
        """
        This function loads in our pre-trained GAN .pb file to generate fake images
        """
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # fix nodes
        for node in graph_def.node:
          if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
              if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
          elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                op_dict=None,
                                producer_op_list=None,
                                name=model_name)

        return graph, graph_def
    
    # zz2517
    @staticmethod
    def load_layers(graph, modelfilename, model_name, gen_input, gen_output, gen_loss, disc_input, disc_output):
        """
        This function loads the required layers from self.graph
        gi: generator input
        go: generator output
        gl: generator loss
        di: discriminator input
        do: discriminator output
        """
        if modelfilename == 'graphs/dcgan-100.pb':
            go = graph.get_tensor_by_name(model_name + '/' + gen_output)
            do = graph.get_tensor_by_name(model_name + '/' + disc_output)

        else:
            go = graph.get_tensor_by_name(model_name + '/' + 'generator/' +
                                               gen_output)
            do = graph.get_tensor_by_name(model_name + '/' +
                                               'discriminator/' + disc_output)

        gi = graph.get_tensor_by_name(model_name + '/' + gen_input)

        gl = graph.get_tensor_by_name(model_name + '/' + gen_loss)
        di = graph.get_tensor_by_name(model_name + '/' + disc_input)

        return gi, go, gl, di, do

    @staticmethod
    def imtransform(img):
        """
        This function rescales the pixel values into range [-1, 1]
        """
        return np.array(img) / 127.5-1

    @staticmethod
    def iminvtransform(img):
        """
        This function rescales the pixel values into range [0, 1]
        """
        return (np.array(img) + 1.0) / 2.0

    # zz2517
    @staticmethod
    def genBinarizeMask(mask, dtype=np.float32):
        """
        This function rescale the values in mask according to data type require.
        dtype = float32: return rescale mask with values in range [0,1]
        dytpe = uint8: return resclae mask with values in range [0, 255]

        """
        assert (np.dtype(dtype) == np.float32 or np.dtype(dtype) == np.uint8)
        bmask = np.array(mask, dtype=np.float32)
        bmask[bmask > 0] = 1.0
        bmask[bmask <= 0] = 0
        if dtype == np.uint8:
            bmask = np.array(bmask * 255, dtype=np.uint8)
        return ModelInpaint.duplicateMask(bmask)

    # zz2517
    @staticmethod
    def genWeightedMask(input_mask, batch_size, window_size=7):
        """
        This function initialized weighted mask W with given input mask.
        mask: input mask numpy float32 array
        nsize: window size used to count neighborhood
        """
        kernel = np.ones([window_size, window_size], dtype=np.float32)
        kernel = kernel / np.sum(kernel)
        weighted_mask = input_mask * convolve2d(input_mask, kernel, mode='same', boundary='symm')

        mask = ModelInpaint.duplicateMask(weighted_mask);

        return np.repeat(mask[np.newaxis, :, :, :], batch_size, axis=0)

    # zz2517 wb2328
    @staticmethod
    def duplicateMask(mask):
        """
        create 3D mask by making copies of input 2D mask
        return: a 3D mask with shape (mask.height, mask.width, 3)
        """
        assert(len(mask.shape)==2)
        return np.repeat(mask[:,:,np.newaxis], 3, axis=2)

    @staticmethod
    def poissonblending(img1, img2, mask):
        """Helper: interface to external poisson blending"""
        return blending.blend(img1, img2, 1 - mask)
