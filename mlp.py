from tensorflow import keras
import numpy as np
import json
from pathlib import Path

from trainingCallback import TrainingCallback


class MLPModel:
    def __init__(self, train_new=True, filename='model.h5', batch_size=20, epochs=200, verbose=0):
        self._filename = filename
        self._batch_size = batch_size
        self._epoches = epochs
        self._verbose = verbose

        # self._model = keras.models.Sequential()
        # self._model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
        # self._model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
        # self._model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # self._model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'recall', 'precision'])

        self._model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self._model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        if train_new:
            self.load_weight()

    def getData(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Reserve 10,000 samples for validation
        # x_val = x_train[-10000:]
        # y_val = y_train[-10000:]
        # x_train = x_train[:-10000]
        # y_train = y_train[:-10000]

        return x_train, y_train, x_test, y_test

    def __sample_generator(self, x_train, y_train, batch_size):

        i = 0
        while True:
            x_batch = np.ones(shape=(batch_size, 28, 28))
            y_batch = np.ones(shape=(batch_size, 1))

            total_size = len(x_train)

            for idx in range(batch_size):
                num = np.random.randint(0, total_size)
                x_batch[idx] = x_train[num]
                y_batch[idx] = y_train[num]
            # print(f"[{i}]generator called")
            i += 1
            yield x_batch, y_batch

    def train(self, epochs=1000, eval=100):
        batch_size = 10
        x_train, y_train, x_test, y_test = self.getData()

        gnerator = self.__sample_generator(x_train, y_train, batch_size)
        # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
        #                           write_graph=True, write_images=False)
        steps_per_epoch = len(x_train) / batch_size

        print("batch size: ", batch_size)
        print("steps_per_epoch: ", steps_per_epoch)

        trainingCallback = TrainingCallback('./logs')

        for i in range(epochs):
            history = self._model.fit_generator(gnerator,
                                                steps_per_epoch=3,
                                                epochs=5,
                                                verbose=0,
                                                workers=1,
                                                use_multiprocessing=False,
                                                callbacks=[trainingCallback])

            # print(f"[{i}] loss: ", history.history['loss'], "accuracy: ", history.history['accuracy'])

            if (i % eval) == 0:
                results = self._model.evaluate(x_test, y_test, batch_size=128)
                print(f'[{i}] test loss, test acc:', results)

                status = {
                    'epoch': i,
                    'loss': str(results[0]),
                    'acc': str(results[1])
                }

                with open('status.json', 'w') as f:
                    json.dump(status, f)

                self.save_weight()

    def serv(self):
        return # self._model.predict_classes(data)


    def load_weight(self):
        weights = Path(self._filename)
        if weights.exists():
            self._model.load_weights(self._filename)

    def save_weight(self):
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
