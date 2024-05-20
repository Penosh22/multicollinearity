import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    # MinMax scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # Calculate VIF on scaled data
    vif = calculate_vif(X_scaled)
    st.write(vif)

    # Fit a linear regression model on scaled data
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()

    # Display model summary
    st.text(model.summary())

    # Predictions and intervals
    y_pred = model.predict(X_scaled)
    se = model.bse

    # Calculate confidence intervals
    conf_int = model.conf_int(alpha=0.05)
    conf_int.columns = ['Lower Bound', 'Upper Bound']

    st.write('Confidence Intervals of the Coefficients:')
    st.write(conf_int)

    # Calculate prediction intervals
    predictions = pd.DataFrame({
        'Prediction': y_pred,
        'Lower Bound': y_pred - 1.96 * se.mean(),
        'Upper Bound': y_pred + 1.96 * se.mean()
    })
    
    st.write('Predictions with Prediction Intervals:')
    st.write(predictions)
else:
    st.write("Please select at least one feature.")
