import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

stock_symbol = 'TSLA'
data = yf.download(stock_symbol, start='2015-01-01', end='2023-12-31')

print('first 5 rows of the data:')
print(data.head())
print('\nDescriptive statistics:')
print(data.describe())

print('\nMissing values:')
print(data.isnull().sum())

plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.title(f'{stock_symbol} Historical closing price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

data['next_close'] = data['Close'].shift(-1)
data = data.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['next_close']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('\nTraining set shape:', X_train.shape)
print('Test set shape:', X_test.shape)
print('\nData and scaler saved to .pkl files')