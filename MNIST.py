# Downloading the Mnist Data
# The MNIST dataset is one of the most common datasets used for image classification and accessible from many different sources.
# In fact, even Tensorflow and Keras allow us to import and download the MNIST dataset directly from their API.
# Therefore, I will start with the following two lines to import tensorflow and MNIST dataset under the Keras API.

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# The MNIST database contains 60,000 training images and 10,000 testing images taken from American Census Bureau employees and American high school students [4].
# Therefore, in the second line, I have separated these two groups as train and test and also separated the labels and the images.
# x_train and x_test parts contain greyscale RGB codes (from 0 to 255) while y_train and y_test parts contains labels from 0 to 9 which represents which number they actually are.
# To visualize these numbers, we can get help from matplotlib.

import matplotlib.pyplot as plt
image_index = 7777
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')

#We also need to know the shape of the dataset to channel it to the convolutional neural network.
# Therefore, I will use the “shape” attribute of numpy array with the following code:
print(x_train.shape)

# Reshaping and Normalizing the Images
# To be able to use the dataset in Keras API, we need 4-dims numpy arrays. However, as we see above, our array is 3-dims.
# In addition, we must normalize our data as it is always required in neural network models.
# We can achieve this by dividing the RGB codes to 255 (which is the maximum RGB code minus the minimum RGB code).
# This can be done with the following code:

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_train.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#To be able to use the dataset in Keras API, we need 4-dims numpy arrays.
# However, as we see above, our array is 3-dims.
# In addition, we must normalize our data as it is always required in neural network models.
# We can achieve this by dividing the RGB codes to 255 (which is the maximum RGB code minus the minimum RGB code).
# This can be done with the following code:
print("x_train shape:", x_train.shape)
print("num of img in x_train", x_train.shape[0])
print("num of img in x_test", x_test.shape[0])

#In addition, Dropout layers fight with the overfitting by disregarding some of the neurons while training
# while Flatten layers flatten 2D arrays to 1D array before building the fully connected layers.
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

#Finally, you may evaluate the trained model with x_test and y_test using one line of code:
model.evaluate(x_test, y_test)


image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())