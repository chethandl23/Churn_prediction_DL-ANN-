import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

#load trained models and preprocessors
model = tf.keras.models.load_model('churn_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ohe_geo.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)



# Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
# Input fields
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
Geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
Gender = st.selectbox("Gender", options=['Male', 'Female'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
input_data = {
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}

# One-hot encode Geography in a robust way: handle sparse/dense outputs,
# sklearn API differences (get_feature_names_out vs get_feature_names),
# and unknown categories.
try:
    geo_transformed = encoder.transform([[Geography]])
    try:
        # If it's sparse, this works
        geo_array = geo_transformed.toarray()
    except Exception:
        # If it's already a numpy array, fall back to np.asarray
        geo_array = np.asarray(geo_transformed)

    if hasattr(encoder, 'get_feature_names_out'):
        cols = encoder.get_feature_names_out(['Geography'])
    elif hasattr(encoder, 'get_feature_names'):
        cols = encoder.get_feature_names(['Geography'])
    else:
        cols = [f'Geography_{i}' for i in range(geo_array.shape[1])]

    geo_df = pd.DataFrame(geo_array, columns=cols)
except ValueError as e:
    # This can happen when the encoder sees an unknown category.
    st.warning(f"Warning during encoder.transform: {e}")
    geo_df = pd.get_dummies(pd.Series([Geography]), prefix='Geography')
input_df = pd.DataFrame([input_data])
input_df['Gender'] = label_encoder.transform(input_df['Gender'])
input_df = pd.concat([input_df.drop('Geography', axis= 1), geo_df], axis=1)
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]
st.write(f"Churn Probability: {churn_probability:.2f}")
if churn_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
