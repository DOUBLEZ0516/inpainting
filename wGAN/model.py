import tensorflow as tf

# hz2475
def lrelu(x, alpha=0.2):
    """
    leaky ReLU activation function

    :param x: <tf.tensor> input batch
    :param alpha: <float32> leaky slope
    :return: <tf.tensor> output batch
    """
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# hz2475
def sdpadding(x):
    """
    Leaky ReLU with one column padding on right and down of the image
    Use for Deconvolution with originally odd number width and length image
    :param x: <tf.tensor> input batch
    :return: <tf.tensor> output batch
    """
    return lrelu(tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]]))

# hz2475
def sdpadding_sigmoid(x):
    """
    Sigmoid activation with one column padding on right and down of the image
    Use for Deconvolution with originally odd number width and length image
    :param x: <tf.tensor> input batch
    :return: <tf.tensor> output batch
    """
    return tf.sigmoid(tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]]))

# hz2475
def sdpadding_elu_p(x):
    """
    Transformed elu activation with one column padding on right and down of the image
    Use for Deconvolution with originally odd number width and length image
    :param x: <tf.tensor> input batch
    :return: <tf.tensor> output batch
    """
    return tf.nn.elu(tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]]) - 1.) + 1.

# hz2475
def sdpadding_elu(x):
    """
    Normal elu activation with one column padding on right and down of the image
    Use for Deconvolution with originally odd number width and length image
    :param x: <tf.tensor> input batch
    :return: <tf.tensor> output batch
    """
    return tf.nn.elu(tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]]))

# hz2475
def sdpadding_elu_1(x):
    """
    Normal elu activation with only one column padding on right  of the image
    Use for Deconvolution with originally odd number width and length image
    :param x: <tf.tensor> input batch
    :return: <tf.tensor> output batch
    """
    return tf.nn.elu(tf.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]]))

# hz2475
def critic(img, reuse=False):
    """
    critic(discriminator) for wGAN, trained to distinguish generated image with real ones
    :param img: <tensor> input image, generated or real
    :param reuse: <boolean> if reuse parameters
    :return: <float> log probability of weather the image is real or fake
    """
    X_in = img
    with tf.variable_scope("critic", reuse=None) as scope:
        if reuse:
            scope.reuse_variables()
        size = 64 # size of first convolutional layer, times 2 every layer
        X = tf.reshape(X_in, shape=[-1, 218, 178, 3])
        # X: input image
        X = tf.cast(X, tf.float32)
        conv1 = tf.layers.conv2d(
            inputs=X, # [-1, 218, 178, 3]
            filters=size,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.elu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,  # [-1, 109, 89, 64]
            filters=2 * size,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.elu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

        conv3 = tf.layers.conv2d(
            inputs=pool2,  # [-1, 54, 44, 128]
            filters=4 * size,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.elu)
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2, 2],
            strides=2)

        conv4 = tf.layers.conv2d(
            inputs=pool3,  # [-1, 27, 22, 256]
            filters=8 * size,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.elu)
        pool4 = tf.layers.max_pooling2d(
            inputs=conv4,
            pool_size=[2, 2],
            strides=2)
                            # [-1, 13, 11, 512]
        flatten = tf.contrib.layers.flatten(pool4)
        logit = tf.layers.dense(
            inputs=flatten,
            units=1,
            activation=None, )
    return logit            # [-1, 1]

# hz2475
def generator(sample):
    """
    Generator: generator of wGAN, trained to generate image from random number
    :param sample:<tensor> high dimentional vector sampled from gaussian distribution
    :return: generated image
    """
    decode_in = sample # sampled vector
    with tf.variable_scope("generator", reuse=None):
        dense0 = tf.layers.dense(
            inputs=decode_in,
            units=13 * 11 * 512,
            activation=tf.nn.elu)
        unflatten = tf.reshape(dense0, [-1, 13, 11, 512])  # [-1, 13, 11, 512]
        deconv_4 = tf.layers.conv2d_transpose(
            inputs=unflatten,
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=sdpadding_elu_1)
        deconv_3 = tf.layers.conv2d_transpose(
            inputs=deconv_4,  # [-1, 27, 22, 256]
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=tf.nn.elu)
        deconv_2 = tf.layers.conv2d_transpose(
            inputs=deconv_3,  # [-1, 54, 44, 128]
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=sdpadding_elu)
        deconv_1 = tf.layers.conv2d_transpose(
            inputs=deconv_2,  # [-1, 109, 89, 64]
            filters=3,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=tf.nn.elu)

    return deconv_1 # [-1, 218, 178, 3]
