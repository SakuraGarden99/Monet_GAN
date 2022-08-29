import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Getting features from tf_record files of dataset
def list_record_features(tfrecords_path):
    # Dict of extracted feature information
    features = {}
    # Iterate records
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features

filename = "PATH/TO/TF_RECORD.tfrecord"
features = list_record_features(filename)
print(*features.items(), sep='\n')

# Generated using output from previous function
image_feature_description={
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'target' : tf.io.FixedLenFeature([], tf.int64)
}


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def parse_image_function(example_proto):
	"""Only returning images after parsing"""
	example = tf.io.parse_single_example(example_proto, image_feature_description)
	image = decode_image(example["image"])
	return image



def load_dataset(filenames):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(parse_image_function, num_parallel_calls=AUTOTUNE)
  return dataset

monet_ds= load_dataset("MONET/TFRECORD_PATH.tfrecord").batch(1)
photo_ds = load_dataset("PHOTO/TFRECOED_PATH.tfrecord").batch(1)


for sample in monet_ds.take(1):
  plt.imshow(sample[0])
  plt.axis('off')

# or
example_monet_image = next(iter(monet_ds))
example_photo_image = next(iter(photo_ds))

plt.subplot(121)
plt.title('Photo')
plt.imshow(example_monet_image[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet')
plt.imshow(example_photo_image[0] * 0.5 + 0.5)


