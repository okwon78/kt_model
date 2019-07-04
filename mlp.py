import datetime
import shutil

import numpy as np
import json

import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import SGD

from db_manager import DBManager
from trainingCallback import TrainingCallback
import tensorflow.keras.backend as K

print('tf.__version__ ', tf.__version__)


def _y_true(y_true, y_pred):
    return y_true


def _y_pred(y_true, y_pred):
    return y_pred


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall = c1 / c3

    return recall


class MLPModel:

    def __init__(self, train_new=True, batch_size=20, epochs=200, verbose=0):

        self._check_point = "model_weight/cp.ckpt"
        self._batch_size = batch_size
        self._epoches = epochs
        self._verbose = verbose

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)

        self._model = keras.models.Sequential()
        self._model.add(Dense(3, input_dim=7, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(5, input_dim=3, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(3, input_dim=5, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(1, input_dim=3, kernel_initializer='normal', activation='sigmoid'))
        self._model.compile(loss='binary_crossentropy', optimizer=sgd,
                            metrics=['accuracy', f1_score, recall, _y_true, _y_pred])

        self._training_callback = TrainingCallback(self._model, self._check_point, './logs')

    def __sample_generator(self, batch_size, dl_id, dl_name):
        dbManager = DBManager()
        dbManager.init(dl_id, dl_name)

        while True:
            try:
                x_batch = np.ones(shape=(batch_size, 7))
                y_batch = np.ones(shape=(batch_size, 1), dtype=np.int32)

                histories = list()
                labels = list()

                index = 0
                while True:
                    dl_hist_id = np.random.randint(low=dbManager.train_min_index(), high=dbManager.train_max_index())
                    history, label = dbManager.get_train_data(dl_hist_id)

                    if history is None:
                        continue

                    histories.append(history)
                    labels.append(label)

                    index += 1

                    if batch_size == index:
                        break

                for idx, elem in enumerate(histories):
                    x_batch[idx] = elem
                    y_batch[idx] = labels[idx]

                # print('x_batch: ', x_batch)
                yield x_batch, y_batch
            except Exception as e:
                print("Exception: ", e)
                continue

    def train(self, epochs=100, eval=10, dl_id=0, dl_name=""):
        # train_start_timestamp = datetime.datetime.now()
        try:
            batch_size = 100
            dbManager = DBManager()
            dbManager.init(dl_id, dl_name)

            steps_per_epoch = dbManager.get_steps(batch_size)

            print("batch size: ", batch_size)
            print("steps_per_epoch: ", steps_per_epoch)

            x_test, y_test = dbManager.validation_data()

            self._training_callback.set_db_manager(dbManager, epochs)

            log_dir = "./logs"

            try:
                shutil.rmtree(log_dir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            tensorboard_callback = TensorBoard(log_dir=log_dir,
                                               histogram_freq=0,
                                               batch_size=batch_size,
                                               write_graph=True,
                                               write_grads=False,
                                               write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                               embeddings_metadata=None,
                                               embeddings_data=None, update_freq='epoch')

            generator = self.__sample_generator(batch_size, dl_id, dl_name)
            self._model.fit_generator(generator,
                                      steps_per_epoch=5,
                                      epochs=epochs,
                                      verbose=0,
                                      workers=1,
                                      use_multiprocessing=False,
                                      callbacks=[self._training_callback, tensorboard_callback])

        finally:
            dbManager.close()

    def serv(self, dl_id=0, dl_name=""):
        dbManager = DBManager()
        dbManager.init(dl_id, dl_name)

        total = dbManager.get_total()

        if total == 0:
            return

        next_step = int(total / min(100, total))
        count = 0
        progress = 0
        dbManager.set_state_update(progress, 1)

        for num, index in enumerate(range(dbManager.serv_min_index(), dbManager.serv_max_index() + 1)):

            if num > count + next_step:
                count += next_step
                progress += max(int(100 / total), 1)
                dbManager.set_state_update(progress, 2)

            data = dbManager.get_serv_data(index)

            if data is None:
                continue

            result = self._model.predict(np.array([data]))
            classes = np.round(result[0])
            prob = result[0]
            print(f'[{index}]', 'data: ', data, ' result: ', classes, ' prob: ', prob)

            dbManager.update_serv_result(index, int(classes))

        dbManager.set_state_update(100, 3)

    @staticmethod
    def load_status():
        try:
            with open('status.json', 'r') as f:
                status = json.load(f)
            return status
        except:
            return None

    def summary(self):
        return self._model.summary()


if __name__ == '__main__':
    m = MLPModel(train_new=False)

    # x_train, y_train, x_test, y_test = m.getData()
    # prediction = m.serv(x_test)
    #
    # for i in range(20):
    #     print('prediction: ', prediction[i])
    #     print('Y: ', y_test[i])

    # status = m.load_status()
    # print('epoch: ', status['epoch'])
    # print('loss', status['loss'])
    # print('acc', status['acc'])

    # print(m.summary())

    m.train()
