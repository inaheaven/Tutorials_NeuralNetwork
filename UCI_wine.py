import numpy as np

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(3)
#num of wine classes
classifications = 3

#load dataset
dataset = np.loadtxt('./data/wine2.csv', delimiter=",")
#split dataset into sets for testing and training
X = dataset[:, 1:14]
Y = dataset[:, 0:1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

#covert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, classifications)
y_test = keras.utils.to_categorical(y_test-1, classifications)

model = Sequential()
#number of attributes for input_dim
#every layer is closley connected to next layer by dense
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(classifications, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=25, epochs=5000, validation_data=(x_test, y_test))