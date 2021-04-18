import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.title('Normalisation')
    st.header('Note that normalisation is only advised if your data are not proportionate.')
    str1 = 'The scale between my data features does not matter. Normalise them.'
    conf = st.checkbox('Confirm that you want to normalise your features.', str1)
    if conf == str1:
        
