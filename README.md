# Car-Price-Prediction-Using-Machine-Learning-Regression-
This project performs a regression analysis to predict car prices (Amount in Million ₦) using machine learning.
🚗 Car Price Prediction Using Machine Learning (Regression)
📌 1. Project Overview

Since the target variable (Amount) is numeric, this is a Regression problem, not classification.

📊 Dataset Features

The dataset contains the following features:

Location

Maker (Brand)

Model

Year

Colour

Car Type

Distance Driven (Km)

Amount (Million ₦) → Target Variable

🛠️ 2. Import Required Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

📂 3. Load the Dataset
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Car.xlsx')
df.head()

🔍 4. Data Understanding
df.info()
df.describe()

🧹 5. Data Cleaning Process
5.1 Remove Irrelevant Columns

The Model column contains many unique values and was removed to simplify beginner-level learning.

df.drop(columns=['Model'], inplace=True)

5.2 Deduplication
df.duplicated().sum()
df.drop_duplicates(inplace=True)

5.3 Fix Structural Errors

Standardizing text formatting:

for col in ['Location', 'Maker', 'Colour', 'Type']:
    df[col] = df[col].str.strip().str.title()

5.4 Handle Missing Values

Filling missing values in Distance_Km using the median:

df['Distance_Km'] = df['Distance_Km'].fillna(df['Distance_Km'].median())

5.5 Data Validation
df.isnull().sum()
df.info()


After cleaning:

4464 rows

No missing values

Correct data types

📊 6. Exploratory Data Analysis (EDA)
6.1 Car Price Distribution
sns.histplot(df['Amount (Million ₦)'], kde=True)
plt.title('Car Price Distribution')

6.2 Price by Car Type
sns.barplot(x='Type', y='Amount (Million ₦)', data=df)
plt.title('Price by Car Type')

Observation: Brand New cars are generally more expensive.

6.3 Price vs Year
sns.scatterplot(x='Year', y='Amount (Million ₦)', data=df)
plt.title('Car Year vs Price')


Observation: Newer cars tend to cost more.

🤖 7. Feature Encoding

Machine learning models only understand numerical data.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['Location', 'Maker', 'Colour', 'Type']:
    df[col] = le.fit_transform(df[col])

⚖️ 8. Feature Scaling
from sklearn.preprocessing import StandardScaler

X = df.drop('Amount (Million ₦)', axis=1)
y = df['Amount (Million ₦)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

✂️ 9. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

🏗️ 10. Regression Model Training
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

10.1 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

10.2 Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

10.3 Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

10.4 Gradient Boosting Regressor
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

📈 11. Model Evaluation

Evaluation Metrics Used:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

R² Score – Model accuracy level

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = [
    {
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2 Score': r2_score(y_test, lr_pred)
    },
    {
        'Model': 'Decision Tree',
        'MAE': mean_absolute_error(y_test, dt_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'R2 Score': r2_score(y_test, dt_pred)
    },
    {
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2 Score': r2_score(y_test, rf_pred)
    },
    {
        'Model': 'Gradient Boosting',
        'MAE': mean_absolute_error(y_test, gb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'R2 Score': r2_score(y_test, gb_pred)
    }
]

results_df = pd.DataFrame(results)
results_df.sort_values(by='R2 Score', ascending=False)

🚘 12. Predicting Price for a New Car
new_car = pd.DataFrame({
    'Location': [0],
    'Maker': [3],
    'Year': [2018],
    'Colour': [2],
    'Type': [1],
    'Distance_Km': [60000]
})

new_car_scaled = scaler.transform(new_car)
predicted_price = rf.predict(new_car_scaled)

print('Predicted Car Price (Million ₦):', predicted_price[0])


Example Output:

Predicted Car Price (Million ₦): 24.3285

🎓 13. Final Student Project Summary
📌 Project Title

Car Price Prediction Using Machine Learning

🎯 Aim

To predict car prices based on vehicle characteristics using regression models.

🏁 Conclusion

Among all models tested:

Gradient Boosting Regressor provided the best performance.

It can be used effectively to estimate car prices.
