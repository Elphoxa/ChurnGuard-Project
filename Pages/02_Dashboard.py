import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from numerize.numerize import numerize
 
st.set_page_config(
    page_title='Dashboard',
    page_icon='📈',
    layout='wide'
)
 
# Check if the user is authenticated
if not st.session_state.get("authentication_status"):
    st.info('Please log in to access the application from the homepage.')
else:

    data_d = pd.read_csv('./Dataset/Expresso Customer Data.csv', chunksize=100000)


    
    data = next(data_d)
    # Access data from session state
    #data = st.session_state.get("data_key", None)
    churn_rate = (data["CHURN"].sum() / data.shape[0]) * 100
    On_net_calls = data['ON_NET'].sum()
    average_MONTANT_charges = data['MONTANT'].mean()
    total_revenue = data['REVENUE'].sum()
    total_Users = data['user_id'].count()


    if data is not None:
        
           with st.container():
                      
                                #3. columns
                total1,total2,total3,total4,total5 = st.columns(5,gap='small')
                with total1:

                    st.info('On_Net Sum', icon="🔍")
                    st.metric(label = '', value= f"{On_net_calls:,.0f}")
                    
                with total2:
                    st.info('Average_Montant', icon="🔍")
                    st.metric(label='', value=f"{average_MONTANT_charges:,.0f}")

                with total3:
                    st.info('Total Revenue', icon="🔍")
                    st.metric(label= '',value=f"{total_revenue:,.0f}")

                with total4:
                    st.info('No_of_Users', icon="🔍")
                    st.metric(label='',value=f"{total_Users:,.0f}")

                with total5:
                    st.info('Churn Rate', icon="🔍")
                    st.metric(label='',value=numerize(churn_rate),help=f"""Total rating: {churn_rate}""")
    
                st.markdown("""---""")

                      
                unsafe_allow_html=True,
            

                                  
    def main():
        
         
     
       
        with st.container():
                    
                    data_df= next(data_d)
                    data_df.columns.tolist()

                    cpp1, cpp2 = st.columns(2)
                    with cpp1:
                     st.title('Univariate Analysis')
                    with cpp2:
                        #selected_uni_feature = st.selectbox('Select Feature', data_df)
                        selected_feature=st.selectbox('Select a Feature', options=['REGION', 'TENURE','MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                        'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY',
                        'FREQ_TOP_PACK'], key='selected_model')

                    co1, co2 = st.columns(2)
                                   
                    with co1:
                        fig = px.histogram(data, x=selected_feature, color=selected_feature, barmode="group", height=400, width=500)
                        fig.update_yaxes(title_text=None)
                        st.plotly_chart(fig)
 
                    with co2:
                        fig = px.box(data, x=selected_feature, height=400, width=500)
                        st.plotly_chart(fig)
        with st.container():
                   

                    cn1, cn2 = st.columns(2)
                   
                    with cn1:
                        fig = px.histogram(data, x=selected_feature, height=400, width=500)
                        fig.update_yaxes(title_text=None)
                        st.plotly_chart(fig)
 
                    with cn2:
                        senior_citizen_pie = px.pie(data, names=selected_feature, color=selected_feature, title='Churn rate: For Per Selected Features')
                        st.plotly_chart(senior_citizen_pie, use_container_width=True)
                       
        with st.container():
                    cppo1, cppo2, cppo3= st.columns(3)
                    with cppo1:
                     st.title('Bivariate  Analysis')
                    with cppo2:
                        #selected_uni_feature = st.selectbox('Select Feature', data_df)
                        selected_feature1=st.selectbox('Select a Feature', options=['REGION', 'TENURE','MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                        'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY','CHURN',
                        'FREQ_TOP_PACK'], key='selected_modeli')
                    with cppo3:
                        #selected_uni_feature = st.selectbox('Select Feature', data_df)
                        selected_feature2=st.selectbox('Select a Feature', options=['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                        'DATA_VOLUME','REGION', 'TENURE', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY','CHURN',
                        'FREQ_TOP_PACK'], key='selected_modela')
                    c1, c2 = st.columns(2)
               
                    with c1:
                        fig = px.scatter(data, x=selected_feature1, y=selected_feature2, color=selected_feature2,
                 color_discrete_map={'Yes': 'Firebrick', 'No': 'blue'},
                 labels={selected_feature1: selected_feature1, selected_feature2: selected_feature2},
                 title='Scatter Plot',
                 height=400, width=500)

# Update layout
                        fig.update_layout(xaxis_title=selected_feature1, yaxis_title=selected_feature2, showlegend=True)

# Display the scatter plot in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
 
                    with c2:
                                                
                        # Create a pivot table to prepare data for heatmap
                        pivot_data = data.pivot_table(index=selected_feature1, columns=selected_feature2, values='CHURN', aggfunc='count')

                        # Heatmap using Plotly Express
                        fig = px.imshow(pivot_data,
                                        labels={selected_feature1: selected_feature1, selected_feature2: selected_feature2},
                                        color_continuous_scale='blues',
                                        title='Churn Patterns by other Features',
                                        height=400, width=500)

                        # Update layout
                        fig.update_layout(xaxis_title=selected_feature2, yaxis_title=selected_feature1, showlegend=True)

                        # Display the heatmap in Streamlit
                        st.plotly_chart(fig, use_container_width=True)      
                                        
        with st.container():
                    st.header('Multivariate')
                    ca1, ca2 = st.columns(2)
                    df1 = data.drop(columns=['REGION', 'TENURE'])
                   
 
                    with ca1:  
                      correlation_matrix = data.corr(numeric_only=True)
                      plt.figure(figsize=(10, 8))
                      sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                      plt.title("Correlation Matrix")
                      st.pyplot(plt)
                        
 
                    with ca2:
                      correlation_matrix = data.corr(numeric_only=True)
                      plt.figure(figsize=(10, 8))
                      sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                      plt.title("Correlation Matrix")
                      st.pyplot(plt)
          
 
    if __name__ == '__main__':
              main()
 
    
 
          