from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

dftrain = pd.read_csv(
    "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval = pd.read_csv(
    "https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

C_COL = ["sex", "n_siblings_spouses", "parch",
         "class", "deck", "embark_town", "alone"]
N_COL = ["age", "fare"]

fc = []
for fn in C_COL:
    vocab = dftrain[fn].unique()
    fc.append(tf.feature_column.categorical_column_with_vocabulary_list(fn, vocab))

for fn in N_COL:
    fc.append(tf.feature_column.numeric_column(fn, dtype=tf.float32))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=fc)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
print(result)  # evaluation results

results = list(linear_est.predict(eval_input_fn))
print(result)  # actual prediction results
