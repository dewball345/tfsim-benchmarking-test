from collections import defaultdict
import tensorflow_datasets as tfds
from pathlib import Path
import shutil
import os
import tensorflow as tf

from tqdm import tqdm

def clean_dir(fpath):
    "delete previous content and recreate dir"
    dpath = Path(fpath)
    if dpath.exists():
        shutil.rmtree(fpath)
    dpath = dpath.mkdir(parents=True)

def serialize_example(x, y):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        
        blist = tf.train.BytesList(value=[value])
        return tf.train.Feature(bytes_list=blist)

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # serialized_ex = tf.io.serialize_tensor(x)
    # print(len(serialized_ex.numpy()))
    feature = {
        'x': _bytes_feature(x),
        'y': _int64_feature(y)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def resize(img, size):
    with tf.device("/cpu:0"):
        return tf.image.resize_with_pad(img, size, size)

def parse_image_function(example_proto):
    image_feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto,
                                         image_feature_description)

    parsed_image = tf.io.decode_png(example["x"], channels=3)
    parsed_label = tf.cast(example['y'], tf.int32)

    return parsed_image / 255, parsed_label