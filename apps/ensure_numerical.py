import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

if True:
#CLASSIFY FEATURES MANUALLY
    numerical_cols = pd.read_csv('numerical_cols.csv')
    categorical_cols = pd.read_csv('categorical_cols.csv')

    #delete numerical_cols, categorical_cols here


    #if st.button('Classify features manually'):
    def classify_features_manually(df):
        global numerical_cols
        global categorical_cols
        st.header('Part I: Check numerical columns')
        st.write("Help us make sure we've classified the features correctly. Meanwhile, we'll try and develop a better algorithm to do so! 😅")

        turn = {}
        for col in df.columns:
            #if col in numerical_cols.columns:
            ftr = st.selectbox(str(col+' is a numerical feature.'), ('Numerical', 'Categorical'))
            if ftr == 'Categorical':
                turn[col] = True

        if st.button('Convert these to categorical'):
            for col in turn:
                if turn[col] == True:
                    st.write('Converting ', col, ' to a categorical column')
                    #categorical_cols.append(numerical_cols[col])
                    categorical_cols[col] = numerical_cols[col] #.astype('object')
                    numerical_cols = numerical_cols.drop([col], axis = 1)

            st.write('These are the numerical columns now:')
            st.write(str(numerical_cols.columns.tolist()))

            st.write('---')
            st.write('and these are categorical columns:')
            st.write(str(categorical_cols.columns.tolist()))


            open('numerical_cols.csv', 'w').write(numerical_cols.to_csv(index = False))  #save numerical_cols to directory
            open('categorical_cols.csv', 'w').write(categorical_cols.to_csv(index = False))  #save categorical_cols to directory
            st.success('Features saved.')


    classify_features_manually(numerical_cols)

    st.write('---')
