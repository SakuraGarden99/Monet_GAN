import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from IPython.display import clear_output

dataset, metadata = tfds.load('cycle_gan/monet2photo', with_info=True, as_supervised=True)

train_monet, train_photo = dataset['trainA'], dataset['trainB']
test_monet , test_photo= dataset['testA'], dataset['testB']

def normalize(image):
  image= tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  return image

def random_augment(image):
  image = tf.image.resize(image, [286,286], method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH,3])
  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_image_train(image):
  image = random_augment(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = normalize(image)
  return image

train_monet = train_monet.map(lambda x,y:x)
train_photo = train_photo.map(lambda x,y:x)

test_monet,test_photo = test_monet.map(lambda x,y:x), test_photo.map(lambda x,y:x)

train_monet = train_monet.map(preprocess_image_train, num_parallel_calls= tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_photo= train_photo.map(preprocess_image_train, num_parallel_calls= tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_monet = test_monet.map(preprocess_image_test, num_parallel_calls= tf.data.AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_photo = test_photo.map(preprocess_image_test, num_parallel_calls= tf.data.AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def downsample(filters, size, apply_norm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = keras.Sequential()
  result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer= initializer, use_bias= False))

  if apply_norm:
    result.add(InstanceNormalization())
  result.add(layers.LeakyReLU())
  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = keras.Sequential()
  result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer= initializer, use_bias= False))
  
  if apply_dropout:
    result.add(layers.Dropout(0.5))

  result.add(layers.LeakyReLU())
  return result


def discriminator_loss(real, generated):
  real_loss = cross_entropy_loss(tf.ones_like(real), real)
  gen_loss = cross_entropy_loss(tf.zeros_like(generated), generated)
  total_loss = real_loss + gen_loss
  return total_loss*0.5

def generator_loss(preds_generated):
  return cross_entropy_loss(tf.ones_like(preds_generated), preds_generated)

def cycle_loss(real_image, cycled_image):
  loss = tf.reduce_mean(tf.abs(real_image-cycled_image))
  return loss*LAMBDA


def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return loss*LAMBDA*0.5

def generate_images(model, sample_photo):
  generated_prediction = model(sample_photo)
  plt.figure(figsize=(12,12))

  titles= ['Input Image', 'Predicted Image']
  plot_images = [sample_photo[0], generated_prediction[0]]

  for i in range(2):
    plt.subplot(1,2,i+1)
    plt.title(titles[i])
    plt.imshow(plot_images[i]*0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  with tf.GradientTape(persistent=True) as tape:
    # Cycled predictions, output of cycle should be same as input provided
    # Let generator_g transform x->y , and generator_f transform y->x
    # discriminator_x takes x and classify, and discriminator_y takes y and classify

    # 1. generator_g takes x, generate fake_y, then generator_f takes fake_y and generated, cycle_x (x,cycle_x should be similar)

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training= True)

    # 2. Same cycle but starting from generator_f

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training= True)

    # 3. Case when input given to generator is same as expected output

    same_y = generator_g(real_y, training=True)
    same_x = generator_f(real_x, training= True)

    disc_pred_real_x = discriminator_x(real_x, training=True)
    disc_pred_fake_x = discriminator_x(fake_x, training=True)

    disc_pred_real_y = discriminator_y(real_y, training=True)
    disc_pred_fake_y = discriminator_y(fake_y, training=True)

    # All outputs generated, calculating lossess from these
    total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)

    gen_g_identity_loss = identity_loss(real_y, same_y)
    gen_f_identity_loss = identity_loss(real_x, same_x)

    gen_g_loss = generator_loss(disc_pred_fake_y)
    gen_f_loss = generator_loss(disc_pred_fake_x)

    total_gen_g_loss = gen_g_loss + gen_g_identity_loss + total_cycle_loss
    total_gen_f_loss = gen_f_loss + gen_f_identity_loss + total_cycle_loss

    disc_x_loss = discriminator_loss(disc_pred_real_x, disc_pred_fake_x)
    disc_y_loss = discriminator_loss(disc_pred_real_y, disc_pred_fake_y)

  gen_g_grads = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
  gen_f_grads = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

  disc_x_grads = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
  disc_y_grads = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

  generator_g_optimizer.apply_gradients(zip(gen_g_grads, generator_g.trainable_variables ))
  generator_f_optimizer.apply_gradients(zip(gen_f_grads, generator_f.trainable_variables ))

  discriminator_x_optimizer.apply_gradients(zip(disc_x_grads, discriminator_x.trainable_variables ))
  discriminator_y_optimizer.apply_gradients(zip(disc_y_grads, discriminator_y.trainable_variables ))

EPOCHS = 10
for epoch in range(EPOCHS):
  for image_x , image_y in tf.data.Dataset.zip((train_photo, train_monet)):
    train_step(image_x, image_y)

  clear_output(wait= True)
  generate_images(generator_g)