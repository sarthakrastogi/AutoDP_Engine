import numpy as np
import pandas as pd
import streamlit as st
import os
def app():

#    st.sidebar.header('Contents/The pipeline')
#    st.sidebar.write('**1. Upload your data**') #home.py
#    st.sidebar.write('2. Dividing numerical and categorical features.') #split_features.py
#    st.sidebar.write('3. Handling outliers') #outliers.py
#    st.sidebar.write('4. Handling missing values') #missing_values.py
#    st.sidebar.write('5. Picking a target feature') #target_feature.py
#    st.sidebar.write('6. Encoding Values') #encoding.py
#    st.sidebar.write('7. Splitting data into train and test sets') #train_test_split_data.py
#    st.sidebar.write('8. Algorithm recommendation') #recommend_algorithm.py
#    st.sidebar.write('9. Picking an algorithm') #select_model.py #includes feature scaling
#    st.sidebar.write('10. Making Predictions') #predict.py

    #-------------------------------------------------------------------------
    st.write(" ")

    #fetch uploaded dataset
    uploaded_df = st.file_uploader('Upload your dataset.', type=['csv'])
    if uploaded_df is not None:
        df_old_name = uploaded_df.name
        uploaded_df.name = 'df.csv'
        with open(uploaded_df.name, "wb") as f:
            f.write(uploaded_df.getbuffer()) #save the uploaded file to directory
        df = pd.read_csv('df.csv')
        df = pd.DataFrame(df)


        st.write("---")
    #-------------------------------------------------------------------


        def fetch_all_cols(dataframe): #fetch a list of all columns
            global cols
            cols = df.columns.tolist()

            #fetch a list of the datatypes of all columns
            dtypes_list = []
            for col in df.columns:
                dtypes_list.append(df[col].dtype)

            fetch_all_cols.cols_df = pd.DataFrame({'Column' : cols,
                                    'Datatype' : dtypes_list },
                                    columns=['Column', 'Datatype'])
        st.write('Preview:')
        st.write(df.head(5))
        fetch_all_cols(df)
        st.write('Processed dataset with ', df.shape[0], ' rows and the following ', df.shape[1], ' columns:')
        st.write(fetch_all_cols.cols_df)


        st.subheader('Next up: 2. classify between numerical and categorical features >>')


    else:
        st.write('Waiting for your dataset. 👀')
        #------------------------------------------------------------------------------------------------------------
