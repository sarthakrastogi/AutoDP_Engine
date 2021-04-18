import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    numerical_cols = pd.read_csv('numerical_cols.csv')
    categorical_cols = pd.read_csv('categorical_cols.csv')

    #pick target variable

    #MAKE NO CHANGES TO numerical_cols AND categorical_cols IN THIS SECTION

    cat_cols2 = categorical_cols.copy()
    num_cols2 = numerical_cols.copy()

    numerical_cols_list = numerical_cols.columns.tolist()
    categorical_cols_list = categorical_cols.columns.tolist()
    cols.reverse()
    st.write("Help us understand which feature you want to predict. We think it's the following:")

    #LET THEM SELECT MULTIPLE FEATURES SARTHAK

    target_feature = st.selectbox("The last feature in your dataset is selected by default.", (cols))
