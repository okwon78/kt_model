import pymssql
import numpy as np
import os


class Record:
    def __init__(self, data, label):
        self.data = data
        self.label = label


class DBManager:

    def __init__(self, server='sql16ssd-003.localnet.kr', db='lime425_rmos', user='lime425_rmos', pwd='rmos0425'):
        self._server = server
        self._db = db
        self._user = user
        self._pwd = pwd
        self._conn = pymssql.connect(self._server, self._user, self._pwd, self._db)
        self._cursor = self._conn.cursor()

        self._min = 0
        self._max = 0
        self._train_min = 0
        self._train_max = 0
        self._total = 0
        self._cache = dict()

        self._x_validation = None
        self._y_validation = None

    def close(self):
        self._conn.close()

    def init(self, dl_id):

        cursor = self._cursor

        cursor.execute(f"SELECT min(DL_HIST_ID) FROM DL_ALARM WHERE DL_ID = {dl_id}")
        self._min = cursor.fetchone()[0]

        cursor.execute(f"SELECT max(DL_HIST_ID) FROM DL_ALARM WHERE DL_ID = {dl_id}")
        self._max = cursor.fetchone()[0]

        self._total = self._max - self._min + 1
        self._train_min = self._min + min(self._total * 0.2, 2000)
        self._train_max = self._max

        self._x_validation = None  # np.ones(shape=(self._train_min - 1, 7), dtype=np.float32)
        self._y_validation = None  # np.ones(shape=(self._train_min - 1, 1), dtype=np.float32)

    def get_steps(self, batch_size):
        return int(self._total / batch_size)

    def train_min_index(self):
        return self._train_min

    def train_max_index(self):
        return self._train_max

    def to_array(self, row):
        input = list()
        input.append(row[0])
        input.append(row[1])
        input.append(row[2])
        input.append(row[3])
        input.append(row[4])
        input.append(0 if row[5] is None else 0)
        input.append(row[6])
        if row[7] is True:
            label = 1
        else:
            label = 0

        return np.array(input), label

    def validation_data(self):
        cursor = self._cursor
        if self._x_validation is not None:
            return self._x_validation, self._y_validation
        else:
            try:

                cursor.execute(
                    f"SELECT TEMPERATURE,HUMIDITY,RAIN,SNOW,VISIBILITY,TIDE,WAVE_HEIGHT,LABELING FROM DL_ALARM WHERE DL_HIST_ID > {self._min} AND DL_HIST_ID < {self._train_min}")
                rows = cursor.fetchall()
                self._x_validation = np.ones(shape=(len(rows), 7))
                self._y_validation = np.ones(shape=(len(rows), 1))

                for idx, row in enumerate(rows):
                    data, label = self.to_array(row)
                    self._x_validation[idx] = data
                    self._y_validation[idx] = label
                    # print(row)
                    # print(data, label)

            except Exception as e:
                print("validation_data: ", e)
                return None, None

            return self._x_validation, self._y_validation

    def get_train_data(self, index):
        cursor = self._cursor
        if index in self._cache.keys():
            record = self._cache[index]
            return record.data, record.label
        else:
            try:
                cursor.execute(
                    f"SELECT TEMPERATURE,HUMIDITY,RAIN,SNOW,VISIBILITY,TIDE,WAVE_HEIGHT,LABELING FROM DL_ALARM WHERE DL_HIST_ID = {index}")
                row = cursor.fetchone()

                if row is None:
                    return None, None

                data, label = self.to_array(row)
                self._cache[index] = Record(data, label)
                # print(row)
                # print(data, label)
            except Exception as e:
                print("get_train_data: ", e)
                return None, None

            return data, label

    def set_dl_info_update(self, dl_id, epoch):
        cursor = self._cursor
        cursor.execute(f"exec RMOSREPORT_DL_UPD_DLINFO_COUNT {dl_id}, {epoch}")
        self._conn.commit()

    def set_dl_info_complete(self, dl_id, epoch):
        cursor = self._cursor
        cursor.execute(f"exec RMOSREPORT_DL_UPD_DLINFO_STATUS {dl_id}, {epoch}")
        self._conn.commit()
