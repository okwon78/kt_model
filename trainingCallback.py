import shutil
import tensorflow as tf
import os

from pathlib import Path

from numpy import nan
from tensorflow import keras


class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, model, check_point, log_dir):
        self.model = model
        self.check_point = check_point

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        self.version = tf.__version__

        if '1.13' in self.version:
            self.train_summary_writer = tf.summary.FileWriter(log_dir)
        else:
            self.train_summary_writer = tf.summary.create_file_writer(log_dir)

        self.dbManager = None
        self.next_step = 0
        self.count = 0
        self.progress = 0
        self.total_epochs = 1

    def set_db_manager(self, dbManager, total_epochs):
        self.dbManager = dbManager
        self.next_step = int(total_epochs / min(100, total_epochs))
        self.total_epochs = total_epochs

    def on_train_begin(self, batch, logs=None):
        print(f"[train begin]")
        self.load_weight()
        self.dbManager.set_state_update(self.progress, 1)
        return

    def on_train_end(self, batch, logs=None):
        print(f"[train end][{batch}] {logs}")

        self.save_weight()
        self.dbManager.set_state_update(100, 3)
        return

    def on_epoch_begin(self, epoch, logs={}):
        # print(f"[epoch begin][{epoch}] {logs}")
        self.load_weight()
        return

    def on_epoch_end(self, epoch, logs={}):
        print(f"[epoch end][{epoch}] {logs}")

        if epoch > self.count + self.next_step:
            self.count += self.next_step
            self.progress += max(int(100 / self.total_epochs), 1)
            self.dbManager.set_state_update(self.progress, 2)
            print(f"[progress] count[{self.count}] progress[{self.progress}]")
        self.save_weight()

    def print_weights(self):
        # for layer in self.model.layers:
        #     weights = layer.get_weights()
        #     print(weights)
        return

    def on_train_batch_begin(self, batch, logs={}):
        # print(f"\t\ton_train_batch_begin {batch}")
        return

    def on_train_batch_end(self, batch, logs={}):
        # print(f"\t\ton_train_batch_end {batch}")
        return

    def load_weight(self):
        weights_file = Path(self.check_point)
        dirName = os.path.dirname(weights_file)
        dirPath = Path(dirName)

        if dirPath.exists():
            print("load_weight")
            self.print_weights()
            self.model.load_weights(self.check_point)

    def save_weight(self):
        print("save_weight")
        self.print_weights()
        self.model.save_weights(self.check_point)
