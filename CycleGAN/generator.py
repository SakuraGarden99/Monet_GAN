def unet_generator(output_channels=3):
  down_stack=[
      downsample(64,4, apply_norm=False),
      downsample(128,4),
      downsample(256,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
  ]

  up_stack = [
      upsample(512, 4, apply_dropout=True),
      upsample(512, 4, apply_dropout=True),
      upsample(512, 4, apply_dropout=True),
      upsample(512, 4),
      upsample(256, 4),
      upsample(128, 4),
      upsample(64, 4),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer= initializer, activation= 'tanh')

  concat = layers.Concatenate()
  inputs= layers.Input(shape=[None, None, 3])

  x= inputs

  skips= []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])


  for up,skip in zip(up_stack, skips):
    x= up(x)
    x= concat([x, skip])

  x= last(x)

  return keras.Model(inputs= inputs, outputs= x)