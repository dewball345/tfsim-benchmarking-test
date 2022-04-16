from collections import defaultdict
import tensorflow_datasets as tfds
from pathlib import Path
import shutil
import os
import tensorflow as tf

from tqdm import tqdm
import utils


# hard coded for now
utils.clean_dir('dataset')
os.mkdir('dataset/train')
os.mkdir('dataset/test')
os.mkdir('dataset/seen_queries')
os.mkdir('dataset/unseen_queries')
os.mkdir('dataset/index')

dict_count = defaultdict(int)

splits = ['train', 'test']

num_training = 0
num_testing = 0
num_seen_queries = 0
num_unseen_queries = 0
num_index = 0

writer_dict_training = {}
writer_dict_testing = {}
writer_dict_seen_queries = {}
writer_dict_unseen_queries = {}
writer_dict_index = {}


def img_augmentation(img_batch, target_img_size=224):
    # random resize and crop. Increase the size before we crop.
    img_batch = tf.keras.layers.RandomCrop(target_img_size, target_img_size)(img_batch)
    # random horizontal flip
    img_batch = tf.image.random_flip_left_right(img_batch)
    return img_batch

# Creates tf record writers to append to tfrecord. Cannot create on the fly as it overrides.
for i in tqdm(range(196)):
    if i <= 196 // 2:
        writer_dict_training[i] = tf.io.TFRecordWriter(f'dataset/train/{i}.tfrecords')
        writer_dict_seen_queries[i] = tf.io.TFRecordWriter(f'dataset/seen_queries/{i}.tfrecords')
    else:
        writer_dict_testing[i] = tf.io.TFRecordWriter(f'dataset/test/{i}.tfrecords')
        writer_dict_unseen_queries[i] = tf.io.TFRecordWriter(f'dataset/unseen_queries/{i}.tfrecords')
    
    writer_dict_index[i] = tf.io.TFRecordWriter(f'dataset/index/{i}.tfrecords')

# merges train, test splits
for split_index, split in enumerate(splits):
    cars_ds = tfds.load("cars196", split=split)
    num_query_shots = 0.0004 * len(cars_ds)
    num_index_shots = 0.0004 * len(cars_ds)

    for item_index, item in enumerate(ds_loader := tqdm(cars_ds)):

        x = item['image']
        y = item['label']

        dict_count[y.numpy()] += 1

        if 0 <= y <= 196 // 2:
            if dict_count[y.numpy()] <= num_query_shots:
                writer = writer_dict_seen_queries[y.numpy()]
                num_seen_queries += 1
            elif dict_count[y.numpy()] <= num_query_shots + num_index_shots: # index gets all classes
                writer = writer_dict_index[y.numpy()]
                num_index += 1   
            else:
                writer = writer_dict_training[y.numpy()]
                num_training += 1
        else:
            if dict_count[y.numpy()] <= num_query_shots:
                writer = writer_dict_unseen_queries[y.numpy()]
                num_unseen_queries += 1
            elif dict_count[y.numpy()] <= num_query_shots + num_index_shots: # index gets all classes
                writer = writer_dict_index[y.numpy()]
                num_index += 1 
            else:
                writer = writer_dict_testing[y.numpy()]
                num_testing += 1

        
        x = utils.resize(x, 360)
        x = img_augmentation(x)
        x = tf.cast(x, tf.uint8)
        x = tf.io.encode_png(x) # lossless compression: 23 GB to 1.9 GB
        tf_example = utils.serialize_example(x, y)
        writer.write(tf_example.SerializeToString())
        
        # count
        ds_loader.set_description(f"tr: {num_training} te: {num_testing} i: {num_index} sq: {num_seen_queries} usq: {num_unseen_queries}")

print(num_training)
print(num_testing)
print(num_seen_queries)
print(num_unseen_queries)
print(num_index)