import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_datasets as tfds          # Not installed now
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from IPython.display import clear_output

class dataset_pipeline():
    def __init__(self) -> None:
        self.image_feature_description = {
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'target' : tf.io.FixedLenFeature([], tf.string)
        }
        self.IMAGE_SIZE= [256,256]
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256

    def decode_image(self, image):
        """Decoding image from tf.string form representation from tfrecord"""
        image = tf.image.decode_jpeg(image, channels=3)
        image = (tf.cast(image, tf.float32)/127.5) -1
        image = tf.reshape(image, [*self.IMAGE_SIZE,3])
        return image
    
    def random_augment(self, image):
        image = tf.image.resize(image, [286,286], method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.random_crop(image, size=[self.IMG_HEIGHT, self.IMG_WIDTH,3])
        image = tf.image.random_flip_left_right(image)
        return image
    
    def parse_image(self, example):        
        parsed_example = tf.io.parse_single_example(example, self.image_feature_description)
        image = self.decode_image(parsed_example["image"])
        image = self.random_augment(image)
        return image

    def get_paths(self, directory_path):
        """directory_path : path to the directory containing all tfrecords
        returns the list of all tfrecords"""
        path_list = []
        for path in os.listdir(directory_path):
            path_list.append(os.path.join(directory_path, path))
        return path_list

    def load_tfrecord_dataset(self, directory_path):
        """directory_path : list of paths of all tfrecords."""
        paths = get_paths(directory_path)
        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.map(self.parse_image, num_parallel_calls= tf.data.AUTOTUNE)
        dataset = dataset.map(self.random_augment, num_parallel_calls= tf.data.AUTOTUNE)
        return dataset

    def plot_sample(self, dataset):
        sample= next(iter(dataset))
        plt.imshow(sample[0]*0.5 + 0.5)
        plt.axis('off')
        plt.show()

    def list_record_features(self, tfrecords_path):
        """tfrecords_path : path of tfrecords"""
        pass