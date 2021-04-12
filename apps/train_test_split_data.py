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


    #splitting the data

    #concatinating numerical and categorical dfs
    df = pd.concat([categorical_cols, numerical_cols], axis = 1)

    #splitting between independent and dependent variables
    y = df[target_feature]
    X = df.drop(target_feature, axis = 1)

    #slider to change test size
    st.write('Define the size of the test set. We recommend a value of 20% for most datasets, 10% for large datasets and 5% for very large datasets.')

    test_size = st.slider('Test set size', 0.01, 0.50, 0.20)

    #train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=test_size)


    numerical_cols.to_csv('numerical_cols.csv')
    categorical_cols.to_csv('categorical_cols.csv')

    st.write('---')
