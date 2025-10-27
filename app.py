import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf
import streamlit as st
import numpy as np

## Loading the model
model = tf.keras.models.load_model('model.h5')


## Loading all pickle file
with open('Label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)


with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamapp
st.title("Customer Churn Predection")
st.markdown("""
###  About This Model
This application uses an **Artificial Neural Network (ANN)** trained on customer data to predict the **likelihood of customer churn** — that is, whether a customer is likely to leave the service.

The model considers various features such as:
- **Credit Score, Age, Tenure, Balance,** and **Estimated Salary**
- **Number of Products**, **Credit Card status**, and **Activity level**
- **Geography** and **Gender**

After processing these inputs, the model produces a **churn probability score (0–1)**.  
A higher score (above 0.5) indicates that the customer is **likely to churn**, while a lower score suggests they are **likely to stay**.

>  This ANN was trained using TensorFlow/Keras and scaled input data via scikit-learn’s `StandardScaler`, ensuring consistent feature normalization for accurate predictions.
""")


geography = st.selectbox('geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography using the same fitted encoder
geo_array = one_hot_encoder_geo.transform(pd.DataFrame({"Geography": [geography]}))
try:
    geo_array = geo_array.toarray()  # older sklearn returns sparse
except AttributeError:
    pass

geo_cols = one_hot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_array, columns=geo_cols)

# Combine numeric + geo dummies  (use a clean name like input_df)
input_df = pd.concat([input_data.reset_index(drop=True),
                      geo_df.reset_index(drop=True)], axis=1)

# Align to the scaler's expected columns (order + presence)
expected_cols = getattr(scaler, "feature_names_in_", None)
if expected_cols is not None:
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

input_data_scaled = scaler.transform(input_df)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba >0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")






