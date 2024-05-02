import streamlit as st
import pandas as pd

# Function to load and cache the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    # Load the dataset from the Dataset folder
    df = pd.read_csv("./Dataset/Expresso_Customer_Data.csv") 
    return df

def main():
    # Load the dataset
    df = load_data()

    # Display the first few rows of the dataset
    st.write(df.head())

if __name__ == '__main__':
    main()