import shutil
import tensorflow as tf
from tensorflow import keras


class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.model = None
        self.params = None
        self.iterCount = 0

        shutil.rmtree(log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, batch, logs=None):
        # print(f"on_train_begin {batch}")
        return

    def on_train_end(self, batch, logs=None):
        # print(f"on_train_end {batch}")
        return

    def on_epoch_begin(self, epoch, logs={}):
        # print(f"\ton_epoch_begin {epoch}")
        return

    def on_epoch_end(self, epoch, logs={}):
        self.iterCount += 1
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', logs['loss'], step=self.iterCount)
            tf.summary.scalar('acc', logs['accuracy'], step=self.iterCount)
        # print(f"\ton_epoch_end {self._iterCount}")
        return

    def on_train_batch_begin(self, batch, logs={}):
        # print(f"\t\ton_train_batch_begin {batch}")
        return

    def on_train_batch_end(self, batch, logs={}):
        # print(f"\t\ton_train_batch_end {batch}")
        return
