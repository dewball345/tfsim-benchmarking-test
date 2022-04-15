TF similarity benchmarking but with TFRecordDatasetSampler.

1) Run `save_tfrecords_classes.py` to generate training, testing, seen queries, unseen queries, and index partitions.
Each folder has tfrecords separated by class
2) Run `train.py` to train and save model
3) Run `evaluate.py` to get evaluation metrics

Problems - cannot see binary accuracy while training due to RAM limits

When evaluating cannot sample from unseen queries, seen queries, or index datasets because of not getting enough classes?
Have just used some code from https://github.com/tensorflow/similarity/blob/kaggle/examples/supervised/kaggle_keras_tuner.ipynb
which does this sampling. Cannot use all of test dataset due to ram limits as well. 