import streamlit as st
import numpy as np
import pandas as pd
import pyodbc
import altair as alt
 
# Define a key for storing the data in the session state
DATA_KEY = "data_key"
 
# Set page configuration
st.set_page_config(
    page_title='View Data',
    page_icon='📊',
    layout='wide')

alt.themes.enable("dark")
 
# Check if the user is authenticated
if not st.session_state.get("authentication_status"):
    st.info('Please log in to access the application from the homepage.')
else:
 
    # Title of the page
    st.title('Expresso Data')

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
if __name__ == '__main__':

   
    # Load data from CSV
    #st.cache_data 
    data_d = pd.read_csv('./Dataset/Expresso Customer Data.csv', chunksize=5000)
    data_df= next(data_d)
    
  
   
    st.session_state[DATA_KEY] = data_df
     # Selectbox to choose the type of features to display
    columns_1, columns_2, columns_3 = st.columns(3)  # create columns to organize/design the select box
    with columns_1:
            selected_feature_type = st.selectbox("Select data features", options=['All Features', 'Numerical Features', 'Categorical Features'],
                                            key="selected_columns")
    with columns_2:
            pass
    with columns_3:
            pass
 
        # Display the selected features
    if selected_feature_type == 'All Features':
            # Show all features if selected
            st.write(data_df)
    else:
            # Show selected type of features
            st.write(select_features(selected_feature_type, st.session_state[DATA_KEY]))