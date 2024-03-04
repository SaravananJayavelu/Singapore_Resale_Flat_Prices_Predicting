import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Setting Webpage Configurations
st.set_page_config(page_icon="⚙",page_title="Singapore Flats Resale Price Predictor", layout="wide")

st.title(':red[Smarty] - :blue[Home Singapore Flats Resale Price Predictor 🚀]')

@st.cache_resource
def load_model():
    model = pickle.load(open(r'C:\GUVI\Code\Project\.venv\Project6_Singapore\Predicted_model.pkl', 'rb'))

    return model

model = load_model()


df = pd.read_csv(r'c:\GUVI\Code\Project\.venv\Project6_Singapore\Final_df.csv')

col1,col2,col3 =  st.columns(3)


col4,col5= st.columns(2)



with col1:
    Rooms = st.selectbox('Select No. of Rooms', options = df['number_of_rooms'].value_counts().index.sort_values())

with col2:
    Storey = st.selectbox('Select No. of Storey', options = df['storey'].value_counts().index.sort_values())

with col3:
    Reamining_Lease = st.selectbox('Select Lease Remaining', options = df['remaining_lease_new'].value_counts().index.sort_values())

with col4:
    Town = st.selectbox('Select The Town', options = df['town_encoded'].value_counts().index.sort_values())

with col5:
    Total_Floor_Area = st.selectbox('Select Floor Area', options = df['floor_area_sqm_sqrt'].value_counts().index.sort_values())



user_df = pd.DataFrame([[Rooms,Storey,Reamining_Lease,Town,Total_Floor_Area]], columns = ['number_of_rooms', 'storey', 'remaining_lease_new', 'town_encoded', 'floor_area_sqm_sqrt'])

submit = st.button('Predict Flats Resale Price')

if submit:
      
    result = model.predict(user_df)
    st.subheader(f"[Predicted Flats Resale Price]: Rs.{result[0]}")

# streamlit run "C:\GUVI\Code\Project\.venv\Project6_Singapore\Singapore_flat_resale_prediction.py"  