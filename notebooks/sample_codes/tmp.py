"""
Data Source: Foreign Exchange Historical Data by Mizuho Bank
URL: https://www.mizuhobank.co.jp/market/historical.html
Currencies: USD, GBP, EUR
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv', sep='\t')

# re-format dataset
df[['USD','GBP', 'EUR']] = df[['USD','GBP', 'EUR']].astype('float32')

# check stats
print(df.describe())

# visualise dataset
dates = np.arange(0, len(df))
plt.plot(dates, df['USD'], label='USD')
plt.plot(dates, df['GBP'], label='GBP')
plt.plot(dates, df['EUR'], label='EUR')
plt.legend(loc='left')
plt.xlabel('time')
plt.ylabel('value')
plt.show()

df['USD_X'] = df['USD']
df['USD_y'] = df['USD'].shift(-1) # shift a column by one, and NaN is placed at last row
df = df[['USD_X' ,'USD_y']][:-1]  # remove NaN from the last row

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(df['USD_X'].values.reshape(len(df['USD_X']),1))
y = scaler.fit_transform(df['USD_y'].values.reshape(len(df['USD_y']),1))
del df # memory management

# split into train and test sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train, X_test, y_train, y_test = X[0:train_size], X[train_size:len(X)], y[0:train_size], y[train_size:len(y)]

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=1)
model.save('USD_pred_model.h5')
del model

# load data
model = load_model('USD_pred_model.h5')
# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(y_train)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(y_test)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# visualise the result
dates = np.arange(0,len(trainPredict))
plt.plot(dates, trainPredict, label='Prediction')
plt.plot(dates, trainY, label='Target')
plt.legend(loc='upper left')
plt.xlabel('time')
plt.ylabel('value')
plt.show()

dates = np.arange(0,len(testPredict))
plt.plot(dates, testPredict, label='Prediction')
plt.plot(dates, testY, label='Target')
plt.legend(loc='upper left')
plt.xlabel('time')
plt.ylabel('value')
plt.show()