#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read the data from the CSV file
df = pd.read_csv('Housing.csv')

# Define features (X) and target (y)
X = df.drop(columns=['price'])  # Drop the target column to get features
y = df['price']  # Target column

# List of categorical columns to be encoded
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(drop='first')

# Bundle preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Create and fit the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Display coefficients
regressor = model.named_steps['regressor']
preprocessor = model.named_steps['preprocessor']
onehot_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_columns = np.append(onehot_columns, X.columns.difference(categorical_cols))

print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
print("Feature names:", all_columns)


# In[ ]:




