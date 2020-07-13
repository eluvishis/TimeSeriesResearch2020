import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

#load CSV
dataset_train = pd.read_csv('/Users/eden/Projects/SHIFTProject/IBM.csv')

training_set = dataset_train.iloc[:, 1:2].values #open column
#print(training_set)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #scales to range 0 to 1
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #makes it 3D

regressor = tf.keras.models.Sequential()

regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(0.2))

regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
regressor.add(tf.keras.layers.Dropout(0.2))

regressor.add(tf.keras.layers.LSTM(units = 50))
regressor.add(tf.keras.layers.Dropout(0.2))

regressor.add(tf.keras.layers.Dense(units = 1)) #units is the dim of the output space

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 

regressor.fit(X_train, y_train, epochs = 15, batch_size = 32) #it is shuffling the data hmm because it only depends on the previous 60 days

#predicting

dataset_test = pd.read_csv('/Users/eden/Projects/SHIFTProject/IBMtest.csv') #the next 24 days
real_stock_price = dataset_test.iloc[:, 1:2].values #just the open 

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #from 60 days before the start of the test dataset
inputs = inputs.reshape(-1,1) #reshapes into a 2d array
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) #makes it into a 3d array

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#plot 

plt.plot(real_stock_price, color = 'black', label = 'IBM Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.show()
