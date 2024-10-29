pip install pandas numpy scikit-learn yfinance requests

# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 3: Define Data Retrieval Functions

# Function to get lobbying data (replace with your API or data source)
def get_lobbying_data():
    url = 'https://api.opensecrets.org/?method=indus&output=json&apikey=YOUR_API_KEY'
    response = requests.get(url)
    data = response.json()
    lobbying_data = pd.DataFrame(data['response']['industries']['industry'])
    lobbying_data['Lobbying Intensity'] = lobbying_data['total'] / lobbying_data['num_companies'] # Sample calculation
    lobbying_data = lobbying_data[['industry_code', 'Lobbying Intensity']]
    return lobbying_data

# Function to get stock data from Yahoo Finance
def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    stock_data = stock_data['Adj Close'].pct_change().mean(axis=1).to_frame('Stock Return')
    return stock_data

# Function to get macroeconomic data (GDP growth, etc.)
def get_macro_data():
    url = 'https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.KD.ZG?format=json'
    response = requests.get(url)
    data = response.json()
    gdp_data = pd.DataFrame(data[1])
    gdp_data['GDP Growth'] = gdp_data['value']
    gdp_data = gdp_data[['date', 'GDP Growth']].set_index('date').sort_index()
    return gdp_data

# Step 4: Data Preprocessing and Merging
def prepare_data(tickers, start_date, end_date):
    lobbying_data = get_lobbying_data()
    stock_data = get_stock_data(tickers, start_date, end_date)
    macro_data = get_macro_data()

    # Merge datasets on date or industry code if applicable
    data = stock_data.merge(macro_data, left_index=True, right_index=True, how='left')
    # Add sample financial features if you have them or create mock data
    data['P/E Ratio'] = np.random.uniform(10, 25, size=len(data))  # Example placeholder
    data['Debt Ratio'] = np.random.uniform(0.1, 0.5, size=len(data))  # Example placeholder

    # Assuming all stocks are from one industry for simplicity (you could match with lobbying data otherwise)
    data['Lobbying Intensity'] = lobbying_data['Lobbying Intensity'].mean()  # Placeholder if single industry
    return data

# Step 5: Model Training and Evaluation
def train_model(data):
    # Feature selection
    features = ['Lobbying Intensity', 'P/E Ratio', 'Debt Ratio', 'GDP Growth']
    target = 'Stock Return'
    
    # Train-test split
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")

    # Feature importance analysis
    feature_importance = model.feature_importances_
    for feature, importance in zip(features, feature_importance):
        print(f"{feature}: {importance:.4f}")

    return model

# Step 6: Putting It All Together
if __name__ == "__main__":
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Fetch and prepare data
    data = prepare_data(tickers, start_date, end_date)

    # Train and evaluate the model
    model = train_model(data)

    # Display data preview
    print("\nData Preview:")
    print(data.head())
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assuming 'data' has been prepared from the previous code
# Uncomment the next line if needed
# data = prepare_data(tickers=['AAPL', 'MSFT', 'GOOGL'], start_date='2022-01-01', end_date='2023-01-01')

# Step 3: Feature Selection and Target Variable
features = ['Lobbying Intensity', 'P/E Ratio', 'Debt Ratio', 'GDP Growth']
target = 'Stock Return'

X = data[features]
y = data[target]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Optional: Feature Importance Analysis
feature_importance = model.feature_importances_
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

# Step 8: Identify Top Stocks by Predicted Return (Simulation)
# Add predictions back to the test data for interpretation
test_data = X_test.copy()
test_data['Actual Return'] = y_test
test_data['Predicted Return'] = y_pred
top_stocks = test_data.sort_values(by='Predicted Return', ascending=False).head(10)

print("\nTop Stocks by Predicted Return:")
print(top_stocks)
