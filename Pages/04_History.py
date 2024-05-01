import streamlit as st
import numpy as np
import pandas as pd
import os
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
 

st.set_page_config(
    page_title='History',
    page_icon=':)',
    layout='wide'
)

# Load configuration from YAML file
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
 
# Initialize Streamlit Authenticator with configuration settings
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
 
# Perform user authentication


st.title("Prediction History")
if not st.session_state.get("authentication_status"):
    st.info('Please log in to access the application from the MainPage.')
else:
       
    # Title of the page
    

     # Function to select features based on type
    def select_features(feature_type, df):
        if feature_type == 'Numerical Features':
            # Filter numerical features
            numerical_df = df.select_dtypes(include=np.number)
            return numerical_df
        elif feature_type == 'Categorical Features':
            # Filter categorical features
            categorical_df = df.select_dtypes(include='object')
            return categorical_df
      
        else:
            # Return the entire DataFrame
            return df


    def display_prediction_history():
        

    # History page to display previous predictions

                
                    csv_path = "./data/Prediction_history.csv"
                    df = pd.read_csv(csv_path)
                    
                    return df
            # Add logout button to sidebar


    if __name__ == "__main__":
        df = display_prediction_history()
        st.dataframe(df)


 
