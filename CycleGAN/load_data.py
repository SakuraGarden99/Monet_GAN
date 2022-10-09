import tensorflow_datasets as tfds

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

dataset, metadata = tfds.load('cycle_gan/monet2photo', with_info=True, as_supervised=True)

def normalize(image):
  image= tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  return image
  