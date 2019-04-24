import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Global Settings
ATTRIBUTES = 9
pd.set_option('display.max_columns', ATTRIBUTES)
np.random.seed(3)

# Loading Dataset
df = pd.read_csv("./data/data_ise.csv", delimiter=",")
df = df.drop(['TL BASED'], 1)
df.columns = [df.iloc[0, :]]
df = df.drop(df.index[0])
df = df.set_index("date")
df=df.astype(float)

# Check Basic Data Structure
# print(df.info())
# print(df.keys())
# print(df.shape)
# print(df.head())

# Data Pre-processing
# Check if Any Missing Values exist
# df.isnull().values.any()

# Check Correlation of Data
# sns.set_palette("cubehelix")
# sns.pairplot(df)
# plt.show()

# plt.figure(figsize=(10, 5))
# df.ISE.plot.line()
# plt.show()

# Splitting Data into Training Set and Test Set
X = df.iloc[:, 1:9]
Y = df.iloc[:, 0]
print(X.head())
print(Y.head())

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X.values, Y.values, cv=kfold, n_jobs=1)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=1)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))