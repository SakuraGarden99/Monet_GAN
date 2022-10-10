import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from IPython.display import clear_output

class Dataset_pipeline:
    def __init__(self, batch_size=1, img_width=256, img_height= 256, buffer_size=1000) -> None:
        self.BATCH_SIZE= batch_size
        self.IMG_WIDTH= img_width
        self.IMG_HEIGHT= img_height
        self.BUFFER_SIZE= buffer_size

    def load_tfds_dataset(self, path, train_x='trainA', train_y='trainB', test_x='testA', test_y='testB'):
        dataset, metadata = tfds.load(path, with_info=True, as_supervised=True)
        train_x, train_y = dataset[train_x], dataset[train_y]
        test_x, test_y = dataset[test_x], dataset[test_y]

        train_x, train_y = train_x.map(lambda x,y : x), train_y.map(lambda x,y:x)
        test_x, test_y = test_x.map(lambda x,y : x), test_y.map(lambda x,y:x)

        train_x = train_x.map(self.preprocess_image_train, num_parallel_calls= tf.data.AUTOTUNE).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        train_y = train_y.map(self.preprocess_image_train, num_parallel_calls= tf.data.AUTOTUNE).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        test_x = test_x.map(self.preprocess_image_test, num_parallel_calls= tf.data.AUTOTUNE).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        test_y = test_y.map(self.preprocess_image_test, num_parallel_calls= tf.data.AUTOTUNE).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
    
        return train_x, train_y, test_x, test_y

    def load_from_directory(self, directory_path):
        """directory_path : path to the directory/folder containing the images """

        image_paths = [os.path.join(directory_path, x) for x in os.listdir(directory_path)]
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(self.decode_jpeg, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        return dataset


    def decode_jpeg(self, image_path):
        """image_path : Path of image file to decode"""
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        return tf.image.resize(image, [self.IMG_HEIGHT, self.IMG_WIDTH])

    def normalize(self, image):
        image= tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        return image

    def random_augment(self, image):
        image = tf.image.resize(image, [286,286], method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.random_crop(image, size=[self.IMG_HEIGHT, self.IMG_WIDTH,3])
        image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_image_train(self,image):
        image = self.random_augment(image)
        image = self.normalize(image)
        return image

    def preprocess_image_test(self,image):
        image = self.normalize(image)
        return image

    def plot_sample(self, dataset,):
        sample= next(iter(dataset))
        plt.imshow(sample[0]*0.5 + 0.5)
        plt.axis('off')
        plt.show()
    

dataset_pipeline = Dataset_pipeline()
