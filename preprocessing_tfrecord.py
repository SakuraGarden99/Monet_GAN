import tensorflow as tf

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

def parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

raw_dataset = tf.data.TFRecordDataset(filename)
parsed_image_dataset = raw_dataset.map(parse_image_function)
print(parsed_image_dataset)



