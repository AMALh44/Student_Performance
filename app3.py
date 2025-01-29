import pickle
import pandas as pd
import numpy as np
import streamlit as st

with open('Linear.pkl','rb') as model_file:
    loaded_model = pickle.load(model_file)
    
    
st.title('Student Performance')


# 'Hours_Studied','Attendance','Access_to_Resources_m','Motivation_Level_m'

# Input fields for the user
Hours_Studied = st.number_input('Hours Studied', min_value=0, max_value=24, value=2)
Attendance = st.number_input('Attendance', min_value=0, max_value=100, value=87)
Access_to_Resources_m = st.selectbox('Access to Resources', ['High', 'Medium', 'Low'])
Motivation_Level_m = st.selectbox('Motivation Level', ['High', 'Medium', 'Low'])


# Prepare the input  data as a dictionary
input_data = {
    'Hours_Studied':Hours_Studied,
    'Attendance':Attendance,
    'Access_to_Resources_m':Access_to_Resources_m,
    'Motivation_Level_m':Motivation_Level_m,
}

# Convert input data to Dataframe
new_data = pd.DataFrame([input_data])


lmh={
    'Low':1,
    'Medium':2,
    'High':3
}

new_data['Access_to_Resources_m'] =new_data['Access_to_Resources_m'].map(lmh)
new_data['Motivation_Level_m'] =new_data['Motivation_Level_m'].map(lmh)


# Load the saved features list
df = pd.read_csv("cleaned.csv")
columns_list = df.columns.to_list()

# Reindex to match the original column order
new_data = new_data.reindex(columns=columns_list, fill_value=0)


# Make predictions
prediction = loaded_model.predict(new_data)

if st.button('Predict'):
    st.write('Predict Score:',prediction[0])