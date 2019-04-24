import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(3)
classifications = 8

#load dataset
dataset = np.loadtxt('./data/prima-indians-diabetes.csv', delimiter=",")
#split dataset into sets for testing and training
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)


model = Sequential()
#number of attributes for input_dim
#every layer is closley connected to next layer by dense
model.add(Dense(50, input_dim=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, epochs=1, validation_data=(x_test, y_test))
scores = model.evaluate(X, Y)
print(model.metrics_names[1], scores[1]*100)

from ann_visualizer.visualize import ann_viz;
ann_viz(model, title="", view=True);
print(model.layers[1].get_config());