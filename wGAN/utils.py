import glob
import os

import tensorflow as tf

Dataset = tf.data.Dataset
Iterator = tf.data.Iterator

# hz2475
def image_parser(image_loc):
    """
    parse image from file

    :param image_loc: <string> location of image file
    :return: <tf.int8 tensor> decoded image to 0-255 array
    """
    image_file = tf.read_file(image_loc)
    image_decoded = tf.image.decode_image(image_file, channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32)
    return (image_decoded / 255.) * 2. - 1. # normalize image data to [-1., 1.]

# hz2475
def data_feeder2(data_folders, image_type, batch_size, test_partition):
    """
    create tensorflow data stream from data folder

    :param data_folders: <string> input data location
    :param image_type: <string> input image type (.jpg, .png, .bmp, ...)
    :param batch_size: <int> batch size of training data
    :param test_partition: <float32> percentage of data is used as test data
    :return: <tf operations> train and testing iterator and initializer
    """
    #     print(data_folder + '/*.png')

    # indexing file list from data folder
    # slicing test set
    data_len = len(glob.glob(os.path.join(data_folders[0] + '/*.' + image_type)))
    train_len = int(data_len * (1 - test_partition))
    test_len = data_len - train_len
    # print training data size and test data size
    print("tr: " + str(train_len) + "  ts: " + str(test_len) + "  tt: " + str(data_len))
    train_paths = [sorted(glob.glob(os.path.join(data_folder + '/*.' + image_type)))[:train_len]
                   for data_folder in data_folders]
    # test
    # transform train file list to tensorflow Dataset
    train_path_tf = [tf.constant(data_path) for data_path in train_paths]
    train_data = Dataset.from_tensor_slices(tuple(train_path_tf))
    train_data = train_data.map(image_parser)
    train_data = train_data.batch(batch_size)
    train_iterator = Iterator.from_structure(train_data.output_types,
                                             train_data.output_shapes
                                             )
    train_init_op = train_iterator.make_initializer(train_data)

    # test
    # transform test file list to tensorflow Dataset
    test_paths = [sorted(glob.glob(os.path.join(data_folder + '/*.' + image_type)))[train_len:]
                  for data_folder in data_folders]

    test_path_tf = [tf.constant(data_path) for data_path in test_paths]
    test_data = Dataset.from_tensor_slices(tuple(test_path_tf))
    test_data = test_data.map(image_parser)
    test_data = test_data.batch(batch_size)
    test_iterator = Iterator.from_structure(test_data.output_types,
                                            test_data.output_shapes
                                            )
    test_init_op = test_iterator.make_initializer(test_data)
    # return iterator for generate next_batch and init_op to re_initialize
    return train_iterator, train_init_op, test_iterator, test_init_op
