#init 
#run command python3 -m streamlit run visualizations.py
import pandas as pd
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

file_path = 'paneldata.csv'
data = pd.read_csv(file_path)

# categorize in small/medium/large based on 33%/66%
rvu_thresholds = data['RVUs'].quantile([0.33, 0.66])

def categorize_rvu(rvu):
    if rvu <= rvu_thresholds[0.33]:
        return 'Small'
    elif rvu <= rvu_thresholds[0.66]:
        return 'Medium'
    else:
        return 'Large'

data['RVU_Category'] = data['RVUs'].apply(categorize_rvu)

#preprocess for random forest
X_rf = data[['Patient_Count', 'Total Appts in 2023']]
y_rf = data['RVU_Category']

#split train/test
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

#run random forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_rf_train, y_rf_train)

#preprocess for multiple reg
data_reg = data.dropna(subset=['Patient_Count', 'Total Appts in 2023', 'RVUs'])
X_reg = data_reg[['Patient_Count', 'Total Appts in 2023']]
y_reg = data_reg['RVUs']

#split train/test
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#run linreg
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)

st.title('RVU Prediction and Visualization')

#input fields
patient_count = st.number_input('Patient Count', min_value=0)
total_appts = st.number_input('Total Appointments in 2023', min_value=0)

if st.button('Predict RVU Category'):
    prediction_rf = rf_classifier.predict([[patient_count, total_appts]])
    st.write(f'Predicted RVU Category: {prediction_rf[0]}')

if st.button('Predict RVU Value'):
    prediction_reg = reg_model.predict([[patient_count, total_appts]])
    st.write(f'Predicted RVU Value: {prediction_reg[0]}')
    
    #actual vs predicted & includes user input
    y_reg_pred = reg_model.predict(X_reg_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_reg_test, y_reg_test, color='blue', alpha=0.5, label='Actual Values')
    plt.scatter(y_reg_test, y_reg_pred, color='red', alpha=0.5, label='Predicted Values')
    plt.scatter([patient_count], prediction_reg, color='green', alpha=0.7, label='User Input Prediction', s=100)
    for i in range(len(y_reg_test)):
        plt.plot([y_reg_test.iloc[i], y_reg_test.iloc[i]], [y_reg_test.iloc[i], y_reg_pred[i]], 'gray', lw=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual RVUs')
    plt.ylabel('Predicted RVUs')
    plt.title('Actual vs Predicted RVUs')
    plt.legend()
    st.pyplot(plt)

