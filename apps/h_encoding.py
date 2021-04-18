import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():

    numerical_cols = pd.read_csv('numerical_cols.csv')
    categorical_cols = pd.read_csv('categorical_cols.csv')

    if os.path.exists('numerical_cols.csv'):
        os.remove('numerical_cols.csv')
    if os.path.exists('categorical_cols.csv'):
        os.remove('categorical_cols.csv')

    #ONE HOT ENCODING

    st.write('The categorical features need to be encoded so the ML algorithms can understand the data they store.')

    cat_encoded_head = None

    #@st.cache(suppress_st_warning=True)
    #def one_hot_encoding(df_cat):
    for col in categorical_cols.columns:
        colname = col
        col_dummies = pd.get_dummies(categorical_cols[col], prefix = colname)
        categorical_cols = pd.concat([categorical_cols, col_dummies], axis=1)
        categorical_cols = categorical_cols.drop([col], axis = 1)

    cat_encoded_head = categorical_cols.head(3)



    st.write('Finished One Hot encoding. Have a look:')
    #one_hot_encoding(categorical_cols)
    cat_encoded_head



    numerical_cols.to_csv('numerical_cols.csv')
    categorical_cols.to_csv('categorical_cols.csv')

    st.write('---')
