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