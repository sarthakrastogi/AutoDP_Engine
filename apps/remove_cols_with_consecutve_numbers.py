import streamlit as st
import numpy as np
import pandas as pd


#    categorical_cols = pd.read_csv('categorical_cols.csv')

def app():
#if True:
    st.title('Removing Features with Consecutive Numbers')



    df = pd.read_csv('numerical_cols.csv')
    import sys
    def isConsecutive(A):
        s = set(A)
        if len(A) != len(s):
            return False
        prev = sys.maxsize
        for curr in sorted(s):
            if prev != sys.maxsize and (curr != prev + 1):
                return False
            prev = curr
        return True

    cols_with_consecutive_numbers = []
    for col in df:
        if isConsecutive(df[col]):
            cols_with_consecutive_numbers.append(col)
    #st.write(cols_with_consecutive_numbers)
    #st.write(df.columns)
    st.write("The following features only contain consecutive numbers, which may be of no use to the model. Here are some values stored in these columns:")
    st.write(df[cols_with_consecutive_numbers].head(5))

    st.write("Select the features to be removed. We recommend you remove them all.")
    cols_to_remove = []
    for i, col_B in zip(range(len(cols_with_consecutive_numbers)), cols_with_consecutive_numbers):
        a = st.checkbox(col_B)
        if a == True:
            cols_to_remove.append(col_B)

    if st.button("Delete these features"):
        df = df.drop(cols_to_remove, axis = 1)
        st.write(df.head(3))



#            numerical_cols.to_csv('numerical_cols.csv')
        #categorical_cols.to_csv('categorical_cols.csv')

    st.write('---')
