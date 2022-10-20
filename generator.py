noise_shape = 100

def generator_model():

    model = keras.Sequential()
    model.add(layers.Dense(8*8*512, use_bias= False, input_shape=(noise_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8,8,512)))

    assert model.output_shape==(None, 8,8,512)

    model.add(layers.Conv2DTranspose(filters= 256, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    assert model.output_shape == (None, 16,16,256)

    model.add(layers.Conv2DTranspose(filters= 128, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    assert model.output_shape == (None, 32,32,128)

    model.add(layers.Conv2DTranspose(filters= 64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    assert model.output_shape == (None, 64,64,64)

    model.add(layers.Conv2DTranspose(filters= 32, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    assert model.output_shape == (None, 128,128,32)

    model.add(layers.Conv2DTranspose(filters= 3, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 256,256,3)
    
    return model