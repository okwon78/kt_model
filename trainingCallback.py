import shutil
import signal

import tensorflow as tf
import os

from pathlib import Path

import math
from tensorflow import keras

import subprocess
import psutil


def check_kill_process(name):
    for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)


def start_tensorboard():
    tensorboard = subprocess.Popen(
        ["/home/rmos/anaconda3/envs/rmos/bin/tensorboard", "--logdir=/home/rmos/Dev/kt_model/logs"],
        stdout=subprocess.PIPE)
    tensorboard.stdout.close()
    return tensorboard


class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, model, check_point, log_dir):
        self.model = model
        self.check_point = check_point

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        self.version = tf.__version__

        self.dbManager = None
        self.next_step = 0
        self.count = 0
        self.progress = 0
        self.total_epochs = 1
        self.tensorboard = None

    def set_db_manager(self, dbManager, total_epochs):
        self.dbManager = dbManager
        self.next_step = int(total_epochs / min(100, total_epochs))
        self.total_epochs = total_epochs

    def on_train_begin(self, batch, logs=None):
        print(f"[train begin]")

        check_kill_process('tensorboard')
        self.tensorboard = start_tensorboard()
        self.load_weight()
        self.dbManager.set_state_update(self.progress, 1)
        return

    def on_train_end(self, batch, logs=None):
        print(f"[train end]")

        self.save_weight()
        self.dbManager.set_state_update(100, 5)
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.load_weight()
        return

    def on_epoch_end(self, epoch, logs={}):

        if math.isnan(logs['_y_pred']):
            print('nan occur')

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
            # print("load_weight")
            self.print_weights()
            self.model.load_weights(self.check_point)

    def save_weight(self):
        # print("save_weight")
        self.print_weights()
        self.model.save_weights(self.check_point)
