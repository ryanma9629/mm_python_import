import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

raw = pd.read_csv('data/hmeq_imp_enc.csv')
col_y = 'BAD'
col_X = raw.drop(col_y, axis=1).columns
X = raw[col_X]
y = raw[col_y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=12, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)