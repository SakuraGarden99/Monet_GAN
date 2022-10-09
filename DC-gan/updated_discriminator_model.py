def updated_discriminator_model():
    model = tf.keras.Sequential()
    
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2D(64, 4, strides=(2, 2), padding='same', kernel_initializer=initializer,
                                     input_shape=[256, 256, 3], use_bias=False))
    
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2D(128, 4, strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False))
    
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2D(256, 4, strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False))
    
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.ZeroPadding2D())
    initializer = tf.random_normal_initializer(0., 0.02)
    
    model.add(layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.ZeroPadding2D())
    
    model.add(layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
    