def discriminator(target= True):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp= layers.Input(shape=[None, None, 3], name='input_image')
  x= inp

  if target:
    tar = layers.Input(shape=[None, None, 3], name='target_image')
    x= layers.Concatenate([inp,tar])

  down1= downsample(64,4, apply_norm=False)(x)
  down2= downsample(128,4)(down1)
  down3= downsample(256, 4)(down2)

  zero_pad1= layers.ZeroPadding2D()(down3)

  conv= layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

  norm1= InstanceNormalization()(conv)

  leaky_relu= layers.LeakyReLU()(norm1)

  zero_pad2= layers.ZeroPadding2D()(leaky_relu)

  last = layers.Conv2D(1, 4, strides= 1, kernel_initializer=initializer)(zero_pad2)

  if target:
    return keras.Model(inputs= [inp, last], outputs= last)
  else:
    return keras.Model(inputs= inp, outputs= last)