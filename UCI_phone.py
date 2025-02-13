import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

np.random.seed(3)
classes = 6

#561 Attributes
X_train = np.loadtxt('./data/UCI HAR Dataset/train/X_train.txt')
Y_train = np.loadtxt('./data/UCI HAR Dataset/train/y_train.txt')
X_test = np.loadtxt('./data/UCI HAR Dataset/test/X_test.txt')
Y_test = np.loadtxt('./data/UCI HAR Dataset/test/y_test.txt')

#convert to classes, one-hot
Y_train = keras.utils.to_categorical(Y_train-1, classes)
Y_test = keras.utils.to_categorical(Y_test-1, classes)

model = Sequential()
model.add(Dense(150, input_dim=561, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5, seed=5))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=50, epochs=1000, validation_data=(X_test, Y_test))

