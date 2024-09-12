import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Final.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Sort the DataFrame by 'Price Date' column
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)

# draw boxplot for Price
sns.boxplot(x=df['Price'])

# remove rows with outliers from the dataset on Price using IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]

sns.boxplot(x=df['Price'])

df

# scale the features
scaler = MinMaxScaler()
df[['Arrivals', 'Precipitation', 'Temp']] = scaler.fit_transform(df[['Arrivals', 'Precipitation', 'Temp']])

df.to_csv('tuning.csv', index=False)

df.isnull().sum()

df

df.columns


# Model Libraries Import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.metrics import r2_score

# Ensure the data is sorted by time
df = df.sort_index()


# Define features and target
FEATURES = ['YEAR', 'DOY', 'Temp', 'Variety_Other']

TARGET = 'Price'

# Split the data into train and test sets using train_test_split
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Prepare training data
X_train = train[FEATURES]
y_train = train[TARGET]

# Prepare testing data
X_test = test[FEATURES]
y_test = test[TARGET]

# Create Decision tree Regressor
reg = RandomForestRegressor(n_estimators = 10,
                            max_depth=3,
                            min_samples_split=10,
                            min_samples_leaf=4,
                            max_features='sqrt',
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=42)

# Fit the model
reg.fit(X_train, y_train)

#predict on test set
y_pred = reg.predict(X_test)

y_train_pred = reg.predict(X_train)

train_residuals = y_train - y_train_pred

test_residuals = y_test - y_pred

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

# Calculate the residuals for the test set
residuals = y_test - y_pred

# Create a residual plot for the test set
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x=range(len(residuals)), y=residuals, color='red', alpha=0.5)

# Set the plot title and labels
ax.set_title('Residual Plot for the Test Set')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')

# Add a horizontal line at zero
ax.axhline(y=0, color='r', linestyle='--')

# Show the plot
plt.show()

# evaluate on test set for rmse, r2, mae, mape
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred)*100)

# Calculate the adjusted R^2 score
adjusted_r2 = 1 - ((1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

print('Adjusted R^2 Score:', adjusted_r2)