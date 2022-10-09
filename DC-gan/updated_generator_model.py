LATENT_DIM= 128

def updated_generator_model():
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*128, use_bias=False, input_shape=(128,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((4,4,128)))

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(128,4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(128,4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(128//2,4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(128//4, 4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(128//8, 4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(layers.Conv2DTranspose(3,4, strides=(2,2), padding= 'same', kernel_initializer= initializer, use_bias= False, activation='tanh'))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    return model