# -*- coding: utf-8 -*-

# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib


# import data
orderbook_data = pd.read_csv('PATH/SPY_2012-06-21_34200000_37800000_orderbook_50.csv') # change based on your actual useage.

def label_and_convert_orderbook_data(orderbook_data, number_of_levels):
    # Rename the columns
    columns = []
    for i in range(1, number_of_levels + 1):
        columns.extend([
            f'Ask_Price_{i}',
            f'Ask_Size_{i}',
            f'Bid_Price_{i}',
            f'Bid_Size_{i}'
        ])
    orderbook_data.columns = columns

    # Convert data types to save memory
    for col in columns:
        if 'Price' in col:
            orderbook_data[col] = orderbook_data[col].astype('float32')
        elif 'Size' in col and orderbook_data[col].notna().all(): # Check for NaN values
            orderbook_data[col] = orderbook_data[col].astype('int32')

    return orderbook_data

sequence_length = 1000  # Adjust as needed

number_of_levels = 50  # Adjust based on your data
orderbook_data_new = label_and_convert_orderbook_data(orderbook_data, number_of_levels)

# normalize / standardize
orderbook_data_new = orderbook_data_new.dropna()
scaler = MinMaxScaler(feature_range=(0, 1)) # initializes scaler
orderbook_data_new_scaled = scaler.fit_transform(orderbook_data_new) # use of scaler
orderbook_data_new = pd.DataFrame(orderbook_data_new_scaled, columns=orderbook_data_new.columns) # converts back to dataframe with same names

class DataGenerator(Sequence):
    def __init__(self, data, sequence_length, batch_size):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.indices = np.arange(len(data) - sequence_length)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = [self.data.iloc[i:i+self.sequence_length].values for i in indices]
        y = [self.data.iloc[i+self.sequence_length-1]['Ask_Price_1'] - self.data.iloc[i]['Ask_Price_1'] for i in indices]
        return np.array(X), np.array(y)

# Split the data into training, validation, and test sets
total_size = len(orderbook_data_new)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)

train_data = orderbook_data_new[:train_size]
val_data = orderbook_data_new[train_size:train_size + val_size]
test_data = orderbook_data_new[train_size + val_size:]


# Create data generators
batch_size = 64

train_generator = DataGenerator(train_data, sequence_length=sequence_length, batch_size=batch_size)
val_generator = DataGenerator(val_data, sequence_length=sequence_length, batch_size=batch_size)
test_generator = DataGenerator(test_data, sequence_length=sequence_length, batch_size=batch_size)
print(test_data)

# Parameters
input_sequence_length = sequence_length  # As per your previous code
features = 200  # As per your previous code
lstm_units = 50  # As per your previous code

# Build the model using the functional API
inputs = Input(shape=(input_sequence_length, features))
lstm_out = LSTM(lstm_units, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(inputs)
lstm_out = LSTM(lstm_units, dropout=0.2, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(lstm_out)
output = Dense(1, activation='linear')(lstm_out)

model = Model(inputs=inputs, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mean_absolute_error', optimizer=optimizer)

# Train the model
model.fit(train_generator, epochs=3, validation_data=val_generator)

# Evaluate the model
loss = model.evaluate(test_generator)
print(f"Test Loss: {loss}")

# Predict on the test set
predictions = model.predict(test_generator)

# Extract true values from the test generator
true_values = []
for i in range(len(test_generator)):
    _, y_batch = test_generator[i]
    true_values.extend(y_batch)

true_values = np.array(true_values)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(true_values, predictions)
mse = mean_squared_error(true_values, predictions)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
