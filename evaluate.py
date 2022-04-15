from collections import defaultdict
from operator import index
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim
import os
import utils
from tqdm import tqdm

def load_model():
    custom_objects = {"Similarity>CircleLoss": tfsim.losses.CircleLoss, "Similarity>PNLoss": tfsim.losses.PNLoss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        path = f"./models"
        full_path = os.path.join(os.path.dirname(__file__), path)

        model = tf.keras.models.load_model(full_path)
        sim_model = tfsim.models.SimilarityModel.from_config(model.get_config())
        return sim_model


def split_dataset(ds):
    return tf.data.Dataset.get_single_element(ds.batch(len(ds)))

def run(config):
    model = load_model()
    loss = tfsim.losses.CircleLoss()
    model.compile(optimizer="adam", loss=loss)

    model.reset_index()

    test_ds = tfsim.samplers.TFRecordDatasetSampler(
        f"dataset\\test\\",
        deserialization_fn=utils.parse_image_function,
        batch_size=1,
        example_per_class=1,
        shards_per_cycle=1,
        shard_suffix="*.tfrecords",
        # num_repeat=0
    ).unbatch()#.shuffle(100)

    ## NOTE: Cannot use index ds because accuracy during calibration is bad (doesn't represent all classes?)
    ## Can't use seen_queries as no way to calibrate using tf.data API
    # seen_queries_ds = NestedTFRecordDataSampler(
    #     f"dataset\\seen_queries\\",
    #     deserialization_fn=utils.parse_image_function,
    #     batch_size=1,
    #     example_per_class=1,
    #     shards_per_cycle=1,
    #     shard_suffix="*.tfrecords",
    #     # num_repeat=0
    # ).unbatch()#.shuffle(100)


    QUERY_PER_CLASS = 2
    query = defaultdict(int)
    x_index, y_index = [], []
    x_query, y_query = [], []
    
    # copied from https://github.com/tensorflow/similarity/blob/kaggle/examples/supervised/kaggle_keras_tuner.ipynb
    for idx in tqdm(test_ds.take(650)):
        img = idx[0].numpy()
        class_idx = idx[1].numpy()
        if query[class_idx] < QUERY_PER_CLASS:
            query[class_idx] += 1
            x_query.append(img)
            y_query.append(class_idx)
        else:
            x_index.append(img)
            y_index.append(class_idx)

    x_index = np.array(x_index)
    y_index = np.array(y_index)
    x_query = np.array(x_query)
    y_query = np.array(y_query)
    # x_in, y_in = split_dataset(index_ds.take(200))
    # x_in, y_in = tfsim.samplers.select_examples(x_in, y_in, list(set(y_in.numpy())), 1)
    # print(y_in)
    model.index(x_index, y_index, data=x_index)

    print("INDEXED WORKS")

    # x_sq, y_sq = split_dataset(seen_queries_ds.take(100))
    # x_sq, y_sq = tfsim.samplers.select_examples(x_sq, y_sq, list(set(y_sq.numpy())), 20)
    model.calibrate(x_query, y_query, calibration_metric="f1",
        matcher="match_nearest",
        extra_metrics=["precision", "recall", "binary_accuracy"],
        verbose=1,)
    
    num_neighbors = 5
    num_samples = 4

    for i in range(num_samples):
        nns = model.single_lookup(x_query[i], k=num_neighbors)
        tfsim.visualization.viz_neigbors_imgs(x_query[i], y_query[i], nns)

    
    

if __name__ == "__main__":
    run(None)