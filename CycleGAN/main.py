from CycleGAN_model import *
from Dataset_pipeline import *

dataset_pipeline = Dataset_pipeline()
path = 'cycle_gan/monet2photo'

train_monet, train_photo, test_monet, test_photo = dataset_pipeline.load_tfds_dataset(path)

model = Model()

model.train(train_monet,train_photo)
