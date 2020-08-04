import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from random import random

def dataset_gen(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0]+i[1]] for i in x])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = dataset_gen(5000, test_size=0.3)
    print(y_train[4])
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser, loss="mse")
    
    model.fit(x_train, y_train, epochs=100)
    
    model.evaluate(x_test, y_test, verbose=1)
    
    pred_inp = np.array([[0.2, 0.1], [0.5, 0.3]])
    print(model.predict(pred_inp))