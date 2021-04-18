import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():

        #if model is not None:
    st.write('Pick the values for your independent variables to make a prediction.')



    num_feature_values = []
    cat_feature_values = []
    #RESOLVE UNDO THE ENCODING, MAYBE CHOOSE X FROM UNENCODED COLUMN
    for col in num_cols2:
            #st.write(col)
            #feature_values[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        col_value = st.number_input(col, format="%.4f", step=0.001)
        #st.write(feature_values)
        num_feature_values.append(col_value)
    num_feature_values

    for col in cat_cols2:
            #st.write(col)
            #values =
        col_value = st.radio(col, cat_cols2[col].unique())
        cat_feature_values.append(col_value)
    cat_feature_values



    num_feature_values = pd.Series(num_feature_values)
    cat_feature_values = pd.Series(cat_feature_values)

    feature_values = pd.concat([cat_feature_values, num_feature_values])
    feature_values


    prediction = model.predict(feature_values)



    st.write('The predicted value for ', target_feature, ' is ', prediction)
