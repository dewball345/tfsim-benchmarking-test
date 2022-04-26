import argparse
import json
import os
import tensorflow_similarity as tfsim
import tensorflow as tf
import utils

class BenchmarkTrainer:
    def __init__(self, root_dir, config):
        self.config = config
        self.root_dir = root_dir
        # self.create_ds()

    def create_ds(self):
        def split_dataset(ds):
            return tf.data.Dataset.get_single_element(ds.batch(len(ds)))

        self.train_ds = tfsim.samplers.TFRecordDatasetSampler(
            f"{self.root_dir}\\train\\",
            deserialization_fn=utils.parse_image_function,
            batch_size=64,
            example_per_class=8,
            shards_per_cycle=16,
            shard_suffix="*.tfrecords"
        )

        self.test_ds = tfsim.samplers.TFRecordDatasetSampler(
            f"{self.root_dir}\\test\\",
            deserialization_fn=utils.parse_image_function,
            batch_size=32,
            example_per_class=8,
            shards_per_cycle=16,
            shard_suffix="*.tfrecords"
        )

        # Cannot use SplitValidationLoss due to RAM
        # self.queries_x, self.queries_y = split_dataset(self.test_ds.take(2))
        # self.queries_y = tf.cast(self.queries_y, tf.int32)
        # self.targets_x, self.targets_y = split_dataset(self.test_ds.skip(2).take(1))
        # self.targets_y = tf.cast(self.targets_y, tf.int32)
 
    def build_model(self):
        embedding_size = 128 

        # building model
        self.model = tfsim.architectures.EfficientNetSim(
            [224, 224, 3],
            embedding_size,
            variant="B0", 
            pooling="gem",    # Can change to use `gem` -> GeneralizedMeanPooling2D
            # gem_p=3.0,        # Increase the contrast between activations in the feature map.
            trainable="full",
        )  

    def build_loss(self):  
        epochs = 20
        loss_type = "circle_loss"
        steps_per_epoch = 100
        val_steps = 10

        if loss_type == "circle_loss":
            LR = 0.0001  
            gamma = 256
            margin = 0.25  


            # init similarity loss
            self.loss = tfsim.losses.CircleLoss(gamma=gamma)
        elif loss_type == "multi_similarity":
            alpha=2
            beta=50
            epsilon=0.1
            lambd=1
            LR = 0.002

            self.loss = tfsim.losses.MultiSimilarityLoss(alpha=alpha, beta=beta, epsilon=epsilon, lmda=lambd)
        elif loss_type == "pn_loss":
            negative_mining="semi-hard"
            LR = 0.001

            self.loss = tfsim.losses.PNLoss(negative_mining_strategy=negative_mining)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=self.loss)

    def train(self):
        epochs = 20
        # loss_type = "multi_similarity"
        steps_per_epoch = 100
        val_steps = 50

        # val_loss = tfsim.callbacks.SplitValidationLoss(
        #     self.queries_x,
        #     tf.cast(self.queries_y, dtype=tf.dtypes.int32),
        #     self.targets_x,
        #     tf.cast(self.targets_y, dtype=tf.dtypes.int32),
        #     metrics=["f1", "binary_accuracy"],
        #     known_classes=list(range(196//2)),
        #     k=1,
        # )

        callbacks = [
            # val_loss,
        ]

        self.history = self.model.fit(
            self.train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.test_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
        )     

    def __call__(self):
        self.create_ds()
        self.build_model()
        self.build_loss()
        self.train()

        return self.model


def run(
    config
):
    #TODO: config iteration
    trainer = BenchmarkTrainer("dataset", config)
    model = trainer()
    tf.keras.models.save_model(model, "models")

if __name__ == "__main__":
    # UNCOMMENT IF RUNNING IN VSCODE
    # print(os.listdir())
    # parser = argparse.ArgumentParser(description="Train model")
    # parser.add_argument("--config", "-c", help="config path")
    # args = parser.parse_args()

    # if not args.config:
    #     parser.print_usage()
    #     quit()
    config = None #json.loads(open(args.config).read())
    run(config)

