import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Importing the training set
dataset_train = pd.read_csv('./data/005930_KS.csv')
training_set = dataset_train.iloc[:, 6:7].values
print(training_set)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)

# Getting the real stock price of 2017
dataset_test = pd.read_csv('./data/005930_KS_TEST.csv')
real_stock_price = dataset_test.iloc[:, 6:7].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
print("dt1")
print(dataset_total)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
print('inputs1')
print(inputs)
inputs = inputs.reshape(-1,1)
print('inputs2')
print(inputs)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 84):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='royalblue', marker='.', linestyle='dashed', label='Actual Price')
plt.plot(predicted_stock_price, color='midnightblue',  marker='.', label='Predicted Price')
plt.title('Stock Price - Samsung Electronics: 005930.KS ')
plt.xlabel('Time')
plt.ylabel('Price in KRW')
plt.legend()
plt.savefig("./data/adam_mse.png")
plt.show()

# Visualising the results
# sns.set_palette("husl")
# sns.lineplot(data=predicted_stock_price)
# sns.lineplot(data=real_stock_price)
# plt.show()
# predicted_stock_price[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(12,5))
#
# plt.plot(real_stock_price, color = 'red', label = 'Actual Samsung Stock Price')
# plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Samsung Stock Price')
# plt.title('Samsung Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Samsung Stock Price')
# plt.legend()
# plt.show()