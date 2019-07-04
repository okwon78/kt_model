import pymssql
import numpy as np
from time import gmtime, strftime


class Record:
    def __init__(self, data, label):
        self.data = data
        self.label = label


def to_array_for_train(row):
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


def to_array_for_serv(row):
    input = list()
    input.append(row[0])
    input.append(row[1])
    input.append(row[2])
    input.append(row[3])
    input.append(row[4])
    input.append(0 if row[5] is None else 0)
    input.append(row[6])

    return np.array(input)


class DBManager:

    def __init__(self, server='10.240.10.24', db='rmoshist', user='sa', pwd='102938'):
        self._server = server
        self._db = db
        self._user = user
        self._pwd = pwd
        self._conn = pymssql.connect(host=self._server, port=6387, user=self._user, password=self._pwd, database=self._db)
        self._cursor = self._conn.cursor()

        self._min = 0
        self._max = 0
        self._train_min = 0
        self._train_max = 0
        self._total = 0
        self._cache = dict()

        self._x_validation = None
        self._y_validation = None
        self._dl_id = 0
        self._dl_name = ""

    def close(self):
        self._conn.close()

    def init(self, dl_id, dl_name):

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
        self._dl_id = dl_id
        self._dl_name = dl_name

    def get_steps(self, batch_size):
        return int(self._total / batch_size)

    def get_total(self):
        return self._total

    def serv_min_index(self):
        return self._min

    def serv_max_index(self):
        return self._max

    def train_min_index(self):
        return self._train_min

    def train_max_index(self):
        return self._train_max

    def validation_data(self):
        cursor = self._cursor
        if self._x_validation is not None:
            return self._x_validation, self._y_validation
        else:
            try:

                cursor.execute(
                    f"SELECT TEMPERATURE,HUMIDITY,RAIN,SNOW,VISIBILITY,TIDE,WAVE_HEIGHT,LABELING FROM DL_ALARM "
                    f"WHERE DL_HIST_ID > {self._min} AND DL_HIST_ID < {self._train_min} AND DL_ID={self._dl_id}")
                rows = cursor.fetchall()
                self._x_validation = np.ones(shape=(len(rows), 7))
                self._y_validation = np.ones(shape=(len(rows), 1))

                for idx, row in enumerate(rows):
                    data, label = to_array_for_train(row)
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
                    f"SELECT TEMPERATURE,HUMIDITY,RAIN,SNOW,VISIBILITY,TIDE,WAVE_HEIGHT,LABELING FROM DL_ALARM "
                    f"WHERE DL_HIST_ID = {index} AND DL_ID in (SELECT DL_ID FROM DL_INFO WHERE DL_TYPE=1 AND "
                    f"IS_DELETED_YN = 'N')")
                row = cursor.fetchone()

                if row is None:
                    return None, None

                data, label = to_array_for_train(row)

                for i, num in enumerate(data):
                    if num == None:
                        data[i] = 0

                if label == None:
                    label = 0

                self._cache[index] = Record(data, label)
                # print(row)
                # print(data, label)
            except Exception as e:
                print("get_train_data: ", e)
                return None, None

            return data, label

    def get_serv_data(self, index):
        cursor = self._cursor
        cursor.execute(
            f"SELECT TEMPERATURE,HUMIDITY,RAIN,SNOW,VISIBILITY,TIDE,WAVE_HEIGHT FROM DL_ALARM "
            f"WHERE DL_HIST_ID = {index} AND DL_ID={self._dl_id}")
        row = cursor.fetchone()

        if row is None:
            return None
        else:
            return to_array_for_serv(row)

    def update_serv_result(self, index, result):
        cursor = self._cursor
        cursor.execute(f"UPDATE DL_ALARM SET RESULT={result} WHERE DL_HIST_ID={index} AND DL_ID={self._dl_id}")
        self._conn.commit()

    def set_state_update(self, progress, status):
        cursor = self._cursor
        cursor.execute(f"SELECT * FROM DL_INFO WHERE DL_ID={self._dl_id}")
        row = cursor.fetchone()

        # current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        if row is None:
            cursor.execute(f"INSERT INTO DL_INFO (DL_ID, DL_NAME, DL_STATUS, DL_COUNT, SYS_CREATE_DT) "
                           f"VALUES ({self._dl_id}, '{self._dl_name}', {status}, {progress}, GETDATE())")
        else:
            cursor.execute(f"UPDATE DL_INFO SET DL_STATUS={status}, DL_COUNT={progress}, SYS_UPDATE_DT=GETDATE() "
                           f"WHERE DL_ID={self._dl_id}")

        self._conn.commit()
