TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"

test_fn = lambda: print(TRAIN_URL.split('/')[-1])

test_fn()
