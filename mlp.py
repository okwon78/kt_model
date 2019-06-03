import datetime

import numpy as np
import json
import os

from pathlib import Path
from tensorflow import keras
from tensorflow.python.keras.layers import Dense

from db_manager import DBManager
from trainingCallback import TrainingCallback
import tensorflow.keras.backend as K


def _y_true(y_true, y_pred):
    return y_true


def _y_pred(y_true, y_pred):
    return y_pred


def _c1(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return c1


def _c2(y_true, y_pred):
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return c2


def _c3(y_true, y_pred):
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return c3


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

    def __init__(self, train_new=True, filename='model.h5', batch_size=20, epochs=200, verbose=0):

        self._filename = filename
        self._batch_size = batch_size
        self._epoches = epochs
        self._verbose = verbose

        self._model = keras.models.Sequential()
        self._model.add(Dense(5, input_dim=7, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
        self._model.add(Dense(1, input_dim=5, kernel_initializer='normal', activation='sigmoid'))
        self._model.compile(loss='binary_crossentropy', optimizer='adam',
                            metrics=['accuracy', f1_score, recall, _c1, _c2, _c3, _y_true, _y_pred])

        if train_new:
            self.load_weight()

    def __sample_generator(self, batch_size, dl_id):
        dbManager = DBManager()
        dbManager.init(dl_id)

        while True:
            x_batch = np.ones(shape=(batch_size, 7))
            y_batch = np.ones(shape=(batch_size, 1))

            histories = list()
            labels = list()

            index = 0
            while True:
                dl_hist_id = np.random.randint(low=dbManager.train_min_index(), high=dbManager.train_max_index())
                history, label = dbManager.get_train_data(dl_hist_id)
                # print(history, label)

                if history is None:
                    continue

                histories.append(history)
                labels.append(label)

                index += 1

                if batch_size == index:
                    break

            for idx, histroy in enumerate(histories):
                x_batch[idx] = history
                y_batch[idx] = labels[idx]

            yield x_batch, y_batch

    def train(self, epochs=100, eval=10, dl_id=0):
        # train_start_timestamp = datetime.datetime.now()
        try:
            batch_size = 100
            dbManager = DBManager()
            dbManager.init(dl_id)

            steps_per_epoch = dbManager.get_steps(batch_size)

            print("batch size: ", batch_size)
            print("steps_per_epoch: ", steps_per_epoch)

            trainingCallback = TrainingCallback('./logs')
            x_test, y_test = dbManager.validation_data()

            for i in range(epochs):
                generator = self.__sample_generator(batch_size, dl_id)
                _ = self._model.fit_generator(generator,
                                              steps_per_epoch=5,
                                              epochs=1,
                                              verbose=0,
                                              workers=1,
                                              use_multiprocessing=False,
                                              callbacks=[trainingCallback])

                print(f"[train {i}] loss: ", _.history['loss'], "accuracy: ", _.history['accuracy'], "f1_score",
                      _.history['f1_score'], "recall", _.history['recall'], 'c1:', _.history['_c1'], 'c2:',
                      _.history['_c2'], 'c3:', _.history['_c3'], '_y_pred:', _.history['_y_pred'], '_y_true:',
                      _.history['_y_true'])

                if (i % eval) == 0 and i > 0:
                    results = self._model.evaluate(x_test, y_test, batch_size=128)
                    print(f'[eval {i}]', 'loss:', results[0], 'accuracy:', results[1], 'f1_score', results[2], 'recall',
                          results[3])
                    dbManager.set_dl_info_update(dl_id, i)
                    status = {
                        'epoch': i,
                        'loss': str(results[0]),
                        'acc': str(results[1])
                    }

                    with open('status.json', 'w') as f:
                        json.dump(status, f)

                    self.save_weight()

            dbManager.set_dl_info_complete(dl_id, i)
        finally:
            self.save_weight()
            dbManager.close()

    def serv(self):
        return  # self._model.predict_classes(data)

    def load_weight(self):
        weights_file = Path(self._filename)
        if weights_file.exists():
            self._model.load_weights(self._filename)

    def save_weight(self):
        weights_file = Path(self._filename)
        if weights_file.exists():
            os.remove(self._filename)
        self._model.save_weights(self._filename)

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
