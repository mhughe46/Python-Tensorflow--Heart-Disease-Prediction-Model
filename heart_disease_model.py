import pandas as pd
import tensorflow as tf
import keras as keras
import numpy as np
import keras.layers as layers

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalac', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()

dataset.pop('slope')
dataset.pop('ca')
dataset.pop('thal')


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('num')
test_labels = test_features.pop('num')

normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

heart_model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

heart_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01))

heart_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=2,
    epochs=50
)

heart_model.evaluate(test_features, test_labels)

heart_model.save('heart_model')