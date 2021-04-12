import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.sidebar.header('Contents/The pipeline')
    st.sidebar.write('1. Upload your data') #home.py
    st.sidebar.write('**2. Dividing numerical and categorical features.**') #split_features.py
    st.sidebar.write('3. Handling outliers') #outliers.py
    st.sidebar.write('4. Handling missing values') #missing_values.py
    st.sidebar.write('5. Picking a target feature') #target_feature.py
    st.sidebar.write('6. Encoding Values') #encoding.py
    st.sidebar.write('7. Splitting data into train and test sets') #train_test_split_data.py
    st.sidebar.write('8. Algorithm recommendation') #recommend_algorithm.py
    st.sidebar.write('9. Picking an algorithm') #select_model.py
    st.sidebar.write('10. Making Predictions') #predict.py

    df = pd.read_csv('df.csv')


    #replacing string numbers to numerics
    df = df.replace('zero', 0)
    df = df.replace('one', 1)
    df = df.replace('two', 2)
    df = df.replace('three', 3)
    df = df.replace('four', 4)
    df = df.replace('five', 5)




    #NLP code to convert number names to digits




    #handling values like ?, -, _
    df.replace('?', np.nan, inplace = True)
    df.replace('-', np.nan, inplace = True)
    df.replace('_', np.nan, inplace = True)


    #CHECKING FOR INFINITE values here



    st.write('So, we went through your dataset:')
    st.write(df.head(2))

    #global numerical_cols
    numerical_cols = []
    #global categorical_cols
    categorical_cols = []
    #undecided_cols = []

    def n_c():
        #algorithm to filter numerical and categorical columns
        for col in df.columns:
            #old algorithm
            if df[col].nunique() < df[col].count()/9:
                categorical_cols.append(df[col])
            elif df[col].nunique() >= df[col].count()/9 and df[col].dtype in ['float64','float32','int32','int64']:
                numerical_cols.append(df[col])
            else:
                numerical_cols.append(df[col])

            #new algorithm
    #        if df[col].dtype in ['float64','float32','int32','int64']:
    #            numerical_cols.append(df[col])
    #        elif df[col].nunique() < df[col].count()/9:
    #            categorical_cols.append(df[col])
    #        else:
    #            undecided_cols.append(df[col])

    n_c()

    #PROCESSING NUMERICAL COLUMNS
    numerical_cols = pd.DataFrame(numerical_cols)
    numerical_cols = numerical_cols.T

    #force converting to numerical
    for col in numerical_cols:
        numerical_cols[col] = pd.to_numeric(numerical_cols[col], errors = 'coerce')

    st.write("These are all the numerical columns:")
    #numerical_cols_top_four = numerical_cols.head(2)
    st.write(numerical_cols.head(2))


    #PROCESSING CATEGORICAL COLUMNS
    categorical_cols = pd.DataFrame(categorical_cols)
    categorical_cols = categorical_cols.T
    st.write("And these are the categorical columns:")
    #categorical_cols_top_four = categorical_cols.head(2)
    st.write(categorical_cols.head(2))


    open('numerical_cols.csv', 'w').write(numerical_cols.to_csv(index = False))  #save numerical_cols to directory
    open('categorical_cols.csv', 'w').write(categorical_cols.to_csv(index = False))  #save categorical_cols to directory
    st.write('---')
