import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calculate_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

st.title('Effects of Multicollinearity on OLS Regression')

# Use the provided EUR/USD dataset
df = pd.read_csv('NSEBANK.csv')

# Let the user select the features
features = []
# Add checkboxes for each feature
for feature in ['Open','High','Low','Volume']:
    if st.checkbox(f'Include {feature}'):
        features.append(feature)

# Separate features and target
X = df[features]
y = df['Close']

if len(features) > 0:
    # Calculate VIF
    vif = calculate_vif(X)
    st.write(vif)

    # Fit a linear regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Display model summary
    st.text(model.summary())
else:
    st.write("Please select at least two features.")
