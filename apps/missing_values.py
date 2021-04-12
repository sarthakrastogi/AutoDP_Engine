import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    #handling missing values

    for col in numerical_cols.columns:
        null_count = numerical_cols[col].isnull().sum()
        na_count = numerical_cols[col].isna().sum()
        #nan_count = np.isnan(numerical_cols[col].any()).sum()
    if null_count + na_count==0:
        st.write('Your data has no missing values! Wow, that is rare.')

    else:
        st.write('Your data has missing values.')






        #NOT WORKING:

        #st.write('Have a look:')

        #import missingno as msno
        #missing_bar_fig = msno.bar(uploaded_df)
        #st.write(missing_bar_fig)
        #sno.matrix(uploaded_df)




        #MANUALLY VISUALISE MISSING VALUES IF MSNO DOESNT WORK


        st.write("Missing data introduces bias that leads to misleading results :/")

        cols_with_more_than_25pc_missing = []

        #@st.cache
        def find_more_than_25_pc_missing(df):
            cols_with_more_than_25pc_missing = []
            for col in df:
                if df[col].isnull().sum()/df[col].size >0.25:
                    cols_with_more_than_25pc_missing.append(col)
            if len(cols_with_more_than_25pc_missing) >0:
                st.write("The following columns have more than 25% values missing:")
                for i in cols_with_more_than_25pc_missing:
                    print(i)


        find_more_than_25_pc_missing(numerical_cols)
        find_more_than_25_pc_missing(categorical_cols)

        #switch for whether to impute the missing values in largely empty columns or delete the entire columns

        #@st.cache
        def drop_more_than_25_pc_missing(df):
            if len(cols_with_more_than_25pc_missing) > 0:
                drop_25 = st.radio("It's best to delete these features. What's your opinion?", ("Yes, remove this entire feature.", "No, don't remove the feature. I want to impute these missing values."))
                if drop_25 == "Yes, remove this entire feature.":
                    for col in df.columns:
                        if col in cols_with_more_than_25pc_missing:
                            df.drop([col], axis = 1)
                            st.write('Dropped ', col, ' from the dataframe.')
                else:
                    st.write("Okay, we'll impute these values in the next step.")

            drop_more_than_25_pc_missing(numerical_cols)
            drop_more_than_25_pc_missing(categorical_cols)

            st.write("We can impute the missing values with the mean, median or mode. Pick one. Median is a good default.")

            #switch to choose from imputing with mean, median or mode
            imputer_strategy = st.selectbox("Pick an imputing strategy.", ('Median', 'Mode', 'Mean'))

            impute_num_flag = False
            if st.button('Impute'):
                impute_num_flag = True

        #@st.cache(suppress_st_warning=True)
        def impute_missing_num(num_df):
            if impute_num_flag == True:
                st.write('Now imputing missing values.')
                if imputer_strategy == 'Median':
                #impute with median
                    for col in num_df:
                        #if uploaded_df[col].isnull().any() == True or uploaded_df[col].isna().any() == True:
                        num_df[col].fillna(num_df[col].median(), inplace=True)
                        st.write('Imputed ', col, ' with median.')

                if imputer_strategy == 'Mode':
                #impute with mode
                    for col in num_df:
                        #if uploaded_df[col].isnull().any() == True or uploaded_df[col].isna().any() == True:
                        num_df[col].fillna(num_df[col].mode(), inplace=True)
                        st.write('Imputed ', col, ' with mode.')

                if imputer_strategy == 'Mean':
                #impute with mean
                    for col in num_df:
                        #if uploaded_df[col].isnull().any() == True or uploaded_df[col].isna().any() == True:
                        num_df[col].fillna(num_df[col].mean(), inplace=True)
                        st.write('Imputed ', col, ' with mean.')

        impute_missing_num(numerical_cols)









                #ADD CODE TO IMPUTE CATEGORICAL VALUES
















    #saving imputed def

    numerical_cols.to_csv('numerical_cols.csv')
    categorical_cols.to_csv('categorical_cols.csv')

    st.write('---')
