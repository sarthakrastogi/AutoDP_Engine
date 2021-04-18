import streamlit as st
import numpy as np
import pandas as pd


#    categorical_cols = pd.read_csv('categorical_cols.csv')

def app():
#if True:
    st.title('Handling outliers')


    numerical_cols = pd.read_csv('numerical_cols.csv')

    #TO DO: CHECK FOR OUTLIERS HERE FIRST


    #TO DO: RECOMMEND WHICH LIMIT METHOD AND TREATMENT METHOD TO USE



    iqr_option = 'IQR (Anything above Q1-1.5*IQR and above Q3 + 1.5* IQR is an outlier)'
    ten_ninety_option = 'Below 10th and above 90th quertiles'
    manual_option = 'Manually define limits for each column'
    limit_method = st.radio("How do you want to define your lower and upper limits?", (
                            iqr_option,
                            ten_ninety_option,
                            manual_option)
                            )

    floor_cap_option = "Floor and Cap: Reduce higher outliers to the upper limit, and raise lower outliers to the lower limit"
    trim_option = "Remove the observations containing outliers"
    median_option = "Replace the outliers with the median value of the column"

    treatment_method = st.radio("How do you want to treat your outliers?",(
                                floor_cap_option,
                                trim_option,
                                median_option)
                                )


    if st.button("Treat outliers"):
        for col in numerical_cols.columns: #CHANGE TO FOR COLS IN COLS_WITH_OUTLIERS
            if limit_method == iqr_option:
                IQR = numerical_cols[col].quantile(0.25) - numerical_cols[col].quantile(0.25)
                lower_limit = numerical_cols[col].quantile(0.25) - 1.5 * IQR
                upper_limit = numerical_cols[col].quantile(0.75) + 1.5 * IQR


            if limit_method == ten_ninety_option:
                lower_limit = numerical_cols[col].quantile(0.10)
                upper_limit = numerical_cols[col].quantile(0.90)

            if limit_method == manual_option: #NOT WORKING
                st.write(col)
                IQR = numerical_cols[col].quantile(0.25) - numerical_cols[col].quantile(0.25)
                lower_limit = st.text_input("Lower limit for the column " + col, str(numerical_cols[col].quantile(0.25) - 1.5 * IQR))
                lower_limit = float(lower_limit)
                upper_limit = st.text_input("Upper limit for the column " + col, str(numerical_cols[col].quantile(0.75) + 1.5 * IQR))
                upper_limit = float(upper_limit)
                #st.write("---")



            #Flooring and Capping
            if treatment_method == floor_cap_option:
                numerical_cols[col] = np.where(numerical_cols[col] < lower_limit, lower_limit, numerical_cols[col]) #flooring
                numerical_cols[col] = np.where(numerical_cols[col] > upper_limit, upper_limit, numerical_cols[col]) #capping
                st.write("Raised values below ", lower_limit, " to ", lower_limit, ", and reduced values above ", upper_limit, " to ", upper_limit, " for the column ", col)

            #Trimming
            #for when the dataset is large and the number of outliers is small wrt the dataset size and the
            if treatment_method == trim_option:
                numerical_cols.drop(numerical_cols[(numerical_cols[col] > upper_limit) | (numerical_cols[col] < lower_limit)].index, inplace=True)
                st.write("Removed values below ", lower_limit, " and above ", upper_limit, " from the column ", col)

            #Replacing with median
            if treatment_method == median_option:
                col_median = numerical_cols[col].quantile(0.50)
                numerical_cols[col] = np.where(numerical_cols[col] < lower_limit, col_median, numerical_cols[col])
                numerical_cols[col] = np.where(numerical_cols[col] > upper_limit, col_median, numerical_cols[col])
                st.write("Replaced values below ", lower_limit, " and above ", upper_limit, " with the median value = ", col_median, " in the column ", col)





#    def caps(df):
        #if the highest value has an insanely large number of observations, it's possible that any values higher than that have been bracketed into that bin. This needs resolution.


    numerical_cols.to_csv('numerical_cols.csv')
    #categorical_cols.to_csv('categorical_cols.csv')

    st.write('---')
