# -*- coding: utf-8 -*-
"""LSTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KhIS6Ld02VcELrMXonsoTcyOsjUo7SdH
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Load the data
df = pd.read_csv('Final.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Sort the DataFrame by 'Price Date' column
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)

# remove rows with outliers from the dataset on Price using IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]

# scale the features
scaler = MinMaxScaler()  # or MinMaxScaler()
df[['Arrivals', 'Precipitation', 'Temp']] = scaler.fit_transform(df[['Arrivals', 'Precipitation', 'Temp']])

target = ['Price']
features_selected = ['Arrivals', 'YEAR', 'Variety_Other', 'Precipitation', 'Temp', 'Grade_encoded',
                     'Market Name_Sathyamangalam', 'Market Name_Katpadi(Uzhavar Santhai)']
y = df[target].values
X = df[features_selected].values
#y = df['Price'].values
#X = df.drop('Price', axis=1).values

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
# Define a function to create sequences for LSTM
def create_sequences(X, y, time_steps=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Set time steps for LSTM
time_steps = 50  # You can adjust this value based on your data
# Create sequences
X_seq, y_seq = create_sequences(X, y, time_steps)

#Split whole dataset into train and remainings
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=True, random_state=42)

# Define the LSTM model architecture with multiple LSTM layers
model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=32, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), # droput layer for regularization
        tf.keras.layers.LSTM(units=32, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(units=16, activation='relu'),
        Dropout(0.2), # droput layer for regularization
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='mse')

# Print the model summary
model.summary()

print(X_train.shape)  # Should be (num_samples, timesteps, features)
print(X_test.shape)    # Should be (num_samples, timesteps, features)
print(y_train.shape)  # Should be (num_samples,)
print(y_test.shape)

# early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Convert input data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)

# Predict on the test set
y_pred = model.predict(X_test)

y_train_pred = model.predict(X_train)









# model evaluation on train set
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('Mean Absolute Error:', mean_absolute_error(y_train, y_train_pred))
print('Mean Squared Error:', mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('R2 Score:', r2_score(y_train, y_train_pred))

# model evaluation on test set

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# prompt: Calculate adjusted r2

from sklearn.metrics import r2_score

# Calculate the adjusted R^2 score
adjusted_r2 = 1 - ((1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

print('Adjusted R^2 Score:', adjusted_r2)

# calculate mape using scikit learn

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAPE:", mape)

# Plot the training and validation loss over epochs

import matplotlib.pyplot as plt
# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# plot the model prediction against the actual test values

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
# set title
plt.title('Model predictions vs Actual test data')
plt.show()

# calulate mape on validation set

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAPE:", mape)

# calculate smape using scikit learn

from sklearn.metrics import mean_absolute_percentage_error

smape = mean_absolute_percentage_error(y_test, y_pred)

print("sMAPE:", smape)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Create a prediction error plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(train_residuals, bins=20, alpha=0.5, label='Training Set')
ax.hist(test_residuals, bins=20, alpha=0.5, label='Test Set')
ax.legend()
ax.axvline(0, color='k', linestyle='--', label='Zero Error')
ax.set_title('Prediction Error Plot')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
plt.show()

# prompt: make predictions and plot the predictions of the test set, by reindexing, to date

import pandas as pd
import matplotlib.pyplot as plt
# Make predictions on the test set
y_pred = model.predict(X_test)

# Reindex the predicted values to match the test set dates
y_pred_df = pd.DataFrame(y_pred, index=df.index[-len(y_test):], columns=['Price'])

# Plot the actual and predicted values for the test set
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Actual')
plt.plot(y_pred_df.index, y_pred_df['Price'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Model Predictions vs Actual Test Data')
plt.legend()
plt.show()

# prompt: create a pandas dataframe from numpy array of y_test and y_pred after sorting by orignal date from orignal y_test

import pandas as pd
y_test_df = pd.DataFrame(y_test, index=df.index[-len(y_test):], columns=['Actual_Price'])
y_pred_df = pd.DataFrame(y_pred, index=df.index[-len(y_pred):], columns=['Predicted_Price'])

# Sort both DataFrames by the original date from the y_test DataFrame
y_test_df = y_test_df.sort_index(ascending=True)
y_pred_df = y_pred_df.sort_index(ascending=True)

# Merge the two DataFrames
merged_df = pd.merge(y_test_df, y_pred_df, left_index=True, right_index=True)

# Print the merged DataFrame
print(merged_df)

merged_df

merged_df.to_csv('lstm.csv', index=True)

