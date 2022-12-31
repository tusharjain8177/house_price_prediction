import streamlit as st
import pickle as pt
import pandas as pd
import numpy as np

st.title("House Price Prediction")

pipe = pt.load(open('RidgeModel.pkl', 'rb'))

house_details = pd.read_csv('clean_data.csv')

locations = house_details['location'].unique()
total_sqft = house_details['total_sqft'].values
bath = house_details['bath'].values
bhk = house_details['bhk'].values

col1, col2 = st.columns(2)
with col1:
    selected_location = st.selectbox(
        "Select location from list here",
        locations
    )

with col2:
    selected_bhk = st.text_input("Enter bhk")

col3, col4 = st.columns(2)
with col3:
    no_of_bathroom = st.text_input("Enter Number of bathroom")

with col4:
    sqft = st.text_input("Enter total no of sqft")

input = pd.DataFrame([[selected_location, selected_bhk, no_of_bathroom, sqft]], columns=[
    'location', 'total_sqft', 'bath', 'bhk'])
if st.button('predict Price'):
    prediction = pipe.predict(input)[0]
    prediction = prediction * 1e5
    st.subheader("Price is :" + str(np.round(prediction, 2)))
