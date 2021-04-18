import streamlit as st
import numpy as np
import pandas as pd
import os


def app():
#CLASSIFY FEATURES MANUALLY
    global numerical_cols
    global categorical_cols

    numerical_cols = pd.read_csv('numerical_cols.csv')
    categorical_cols = pd.read_csv('categorical_cols.csv')



    st.write(numerical_cols.head(6))
    #if st.button('Classify features manually'):
    def classify_features_manually(df):

        st.header('Part I: Check numerical columns')
        st.write("Help us make sure we've classified the features correctly. Meanwhile, we'll try and develop a better algorithm to do so! 😅")

        turn = {}
        for col in df.columns:
            #if col in numerical_cols.columns:
            ftr = st.selectbox(str(col+' is:'), ('Numerical', 'Categorical'))
            if ftr == 'Categorical':
                turn[col] = True

        if st.button('Convert these to categorical'):
            for col in turn:
                if turn[col] == True:
                    st.write('Converting ', col, ' to a categorical column')
                    #categorical_cols.append(numerical_cols[col])
                    categorical_cols[col] = df[col] #.astype('object')
                    df = numerical_cols.drop([col], axis = 1)

            st.write('These are the numerical columns now:')
            st.write(str(df.columns.tolist()))

            st.write('---')
            st.write('and these are categorical columns:')
            st.write(str(categorical_cols.columns.tolist()))

            #delete numerical_cols, categorical_cols
            if os.path.exists('numerical_cols.csv'):
                os.remove('numerical_cols.csv')
            if os.path.exists('categorical_cols.csv'):
                os.remove('categorical_cols.csv')

            open('numerical_cols.csv', 'w').write(df.to_csv(index = False))  #save numerical_cols to directory
            open('categorical_cols.csv', 'w').write(categorical_cols.to_csv(index = False))  #save categorical_cols to directory
            st.success('Features saved.')


    classify_features_manually(numerical_cols)



    st.write('---')
