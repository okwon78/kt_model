import pandas as pd
import tensorflow as tf

ALARM_DATA_FILE = './alarm_20181203.csv'

CSV_COLUMN_NAMES = ['Index',
                    'ID',
                    'Latitude',
                    'Longitude',
                    'AlarmLevel',
                    'TypeID',
                    'StartTime',
                    'StartTimeNo',
                    'StopTime',
                    'StopTimeNo',
                    'Criteria',
                    'Position',
                    'Wave',
                    'Humidity',
                    'Rainfall',
                    'Label']


def load_data(y_name='Label'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = ALARM_DATA_FILE
    test_path = ALARM_DATA_FILE

    data = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    data['Label'] = (data['Label'] != 'y').astype(int)

    train_y = data.pop(y_name)
    train_x = data[['Latitude', 'AlarmLevel', 'Longitude', 'TypeID', 'Criteria', 'Position', 'Wave', 'Humidity', 'Rainfall', ]]

    test_x = train_x
    test_y = train_y

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
