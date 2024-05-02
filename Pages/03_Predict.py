import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
import datetime
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
 
# Set Streamlit page configuration
st.set_page_config(
    page_title='Predict',
    page_icon=':)',
    layout='wide'
)

# Check if the user is authenticated
if not st.session_state.get("authentication_status"):
    st.info('Please log in to access the application from the Home Page.')
else:

# Define the LogTransformer class
    class LogTransformer:
        def __init__(self, constant=1):
            self.constant = constant
    
        def transform(self, X_train):
            return np.log1p(X_train + self.constant)
    
    # Define BooleanToStringTransformer class
    class BooleanToStringTransformer(TransformerMixin):
        def fit(self, X, y=None):
            return self
    
        def transform(self, X):
            return X.astype(str)

    # Function to load logistic regression model
    @st.cache_data
    def load_logistic_model():
        model = joblib.load('./Models/logistic_model.joblib')
        return model

    # Function to load random forest model
    @st.cache_data
    def random_forest_model():
        model = joblib.load('./Models/random_forest_model.joblib')
        return model
    
    # Create function to select model
    def select_model():
    
        # create columns to organize/design the select box for selecting model
        columns_1, columns_2, columns_3 = st.columns(3)
        with columns_1:
            # Display select box for choosing model
            st.selectbox('Select a Model', options=['Logistic Model', 'random_forest_model'], key='selected_model')
        with columns_2:
            pass
        with columns_3:
            pass
    
        if st.session_state['selected_model'] == 'Logistic Model':
            pipeline = load_logistic_model()  
        else:
            pipeline = random_forest_model()
    
        return pipeline
    
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None
    
    # if not os.path.exists("./data/Prediction_history.csv"):
            # os.mkdir("./data")
    
    # Create function to make prediction
    def make_prediction(pipeline):
    
        # Extract input features from session state
        region=st.session_state['REGION']
        tenure=st.session_state['TENURE']
        montant=st.session_state['MONTANT']
        frequence_rech=st.session_state['FREQUENCE_RECH']
        revenue=st.session_state['REVENUE']
        arpu_segment=st.session_state['ARPU_SEGMENT']
        frequence=st.session_state['FREQUENCE']
        data_volume=st.session_state['DATA_VOLUME']
        on_net=st.session_state['ON_NET']
        orange=st.session_state['ORANGE']
        tigo=st.session_state['TIGO']
        regularity=st.session_state['REGULARITY']
        freq_top_pack=st.session_state['FREQ_TOP_PACK']
    
        # Define columns and create DataFrame
        columns = [ 'REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 
                'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
        'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY',
        'FREQ_TOP_PACK']
    
        data = [[region, tenure, montant, frequence_rech, revenue,
                arpu_segment,	frequence,	data_volume,	on_net,	
                orange,	tigo,	regularity,	freq_top_pack]]
    
        #Create a Dataframe
        df = pd.DataFrame(data, columns=columns)
    
        df["Date_of_prediction"] = datetime.date.today()
        df["Model"] = st.session_state["selected_model"]
    
        # Make prediction
        predict = pipeline.predict(df)
        prediction = (predict)
    
        # get probability
        probability = pipeline.predict_proba(df)
    
        if prediction == 1:
            df['Predicted Outcome'] = 'Churn'
            df['Probability'] = f'{round(probability[0][1] * 100, 2)}%'
        else:
            df['Predicted Outcome'] = 'Not Churn'
            df['Probability'] = f'{round(probability[0][0] * 100, 2)}%'
        
        df.to_csv("./data/Prediction_history.csv", mode='a', header=not os.path.exists("./data/Prediction_history.csv"), index=False)
    
        # Update session_state with prediction and probability
        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability
    
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
    
        # Store prediction details in session state
        st.session_state.prediction_details = {
            'Prediction': prediction[0],
            'Probability': probability[0],
            'REGION':region, 
            'TENURE':tenure, 
            'MONTANT':montant, 
            'FREQUENCE_RECH':frequence_rech, 
            'REVENUE':revenue,
            'ARPU_SEGMENT':arpu_segment,
            'FREQUENCE':frequence,
            'DATA_VOLUME':data_volume,
            'ON_NET':on_net,
            'ORANGE':orange,
            'TIGO':tigo,
            'REGULARITY':regularity,
            'FREQ_TOP_PACK':freq_top_pack,
        }
    
        return prediction, probability
    
    # Main function
    def main():
    
        pipeline = select_model() # Select model    
    
        # User input form
        with st.form('Features'):
        
            # Create a layout for better organization of the input form
            Column1, Column2, Column3, Column4  = st.columns(4)
    
            # Display input fields
            with Column1:
                # Demographic Information
                st.header('Demographics')
                region= st.selectbox('Select Region', ['DAKAR','SAINT-LOUIS', 'THIES', 'LOUGA', 'MATAM', 'FATICK', 'KAOLACK','DIOURBEL', 'TAMBACOUNDA', 'ZIGUINCHOR', 'KOLDA', 'KAFFRINE', 'SEDHIOU', 'KEDOUGOU'], key='REGION')
                tenure= st.selectbox('Select Tenure', ['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month', 'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'], key='TENURE')
                   
            with Column2:
                # Customer Financial Behaviour
                st.header('Financial Behavior')
                montant=  st.number_input('Enter Top Up Amount ', min_value=20, max_value=500000, key='MONTANT')
                revenue=st.number_input('Monthly Income', min_value=1, max_value=600000, key='REVENUE')
                arpu_segment=st.number_input('Income per 90 Days ', min_value=0, max_value=200000, key='ARPU_SEGMENT')
                
            with Column3:
                # Customer Communication frequency
                st.header('Comm. Frequency')
                frequence_rech=  st.number_input('Refilling Times ', min_value=1, max_value=150, key='FREQUENCE_RECH')
                frequence=st.number_input('Money Making Frequency', min_value=1, max_value=100, key='FREQUENCE')
                regularity=st.number_input('Activeness Count', min_value=0, max_value=70, key='REGULARITY')
                freq_top_pack=st.number_input('Freq Top Up Packages', min_value=0, max_value=700, key='FREQ_TOP_PACK')
            
            with Column4:
                # Customer Usage Section
                st.header('Usage Service')
                data_volume=st.number_input('Data Consumption ', min_value=0, max_value=2000000, key='DATA_VOLUME')
                on_net=st.number_input('Inter Calls Count', min_value=0, max_value=60000, key='ON_NET')
                orange=st.number_input('Calls to Orange', min_value=0, max_value=13000, key='ORANGE')
                tigo=st.number_input('Calls to Tigo', min_value=0, max_value=5000, key='TIGO')
        
            # Submit button to make prediction
            st.form_submit_button ('Predict', on_click=make_prediction, kwargs=dict(
                pipeline=pipeline))
            
    if __name__ == '__main__':
        st.title('Predict Churn')
        main()
    
        prediction = st.session_state['prediction']
        probability = st.session_state['probability']
    
        if not prediction:
            st.markdown('### Prediction would show here ⬇️')
        elif prediction == 1:
            probability_yes = probability[0][1] * 100
            st.markdown(f"### The Customer will Churn with a probability of {round(probability_yes, 2)}%")
        else:
            probability_no = probability[0][0] * 100
            st.markdown(f"### The Customer will not Churn with a probability of {round(probability_no, 2)}%")