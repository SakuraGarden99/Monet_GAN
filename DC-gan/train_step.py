# tf.config.run_functions_eagerly(True)

@tf.function
def train_step(image_batch, train_dim):
    noise = tf.random.normal([1,train_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:            
        generated_image = generator(noise, training=True)

        discriminator_pred_gen = discriminator(generated_image, training=True)

        discriminator_pred_ori= discriminator(image_batch, training=True)


        discriminator_loss = cross_entropy_loss(tf.ones_like(discriminator_pred_ori),discriminator_pred_ori) + cross_entropy_loss(tf.zeros_like(discriminator_pred_gen), discriminator_pred_gen)
        generator_loss = cross_entropy_loss(tf.ones_like(discriminator_pred_gen), discriminator_pred_gen)
        
    generator_grad = gen_tape.gradient(generator_loss, generator.trainable_variables)
    discriminator_grad = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grad,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grad,discriminator.trainable_variables)) 
    
def train(dataset, epochs, train_dim=100):
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            train_step(image_batch, train_dim)
        
        print(f"Time for epoch {epoch+ 1} is : {time.time()-start}")