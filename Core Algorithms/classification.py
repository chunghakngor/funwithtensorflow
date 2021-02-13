from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

# Using Iris Flower dataset
CSV_COL_NAME = ["SepalLength", "SepdalWidth",
                "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginia"]

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COL_NAME, header=0)
test = pd.read_csv(test_path, names=CSV_COL_NAME, header=0)

train_y = train.pop("Species")
test_y = test.pop("Species")


def input_fn(feat, lab, train=True, bs=256):  # features, labels, training, batch_size
    ds = tf.data.Dataset.from_tensor_slices((dict(feat), lab))
    if train:
        ds = ds.shuffle(1000).repeat()
    return ds.batch(bs)


fc = []
for k in train.keys():
    fc.append(tf.feature_column.numeric_column(key=k))

classifier = tf.estimator.DNNClassifier(
    feature_columns=fc, hidden_units=[30, 10], n_classes=3)
classifier.train(input_fn=lambda: input_fn(
    train, train_y, train=True), steps=5000)
result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, train=False))
print(result)  # evaulation results


results = list(classifier.predict(
    input_fn=lambda: input_fn(test, test_y, train=False)))
print(results)  # prediction results
