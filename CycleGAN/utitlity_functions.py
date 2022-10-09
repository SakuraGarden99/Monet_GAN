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

