import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.header('Finding an algorithm that will work best for your dataset...')
    reasons = []

    st.write('Please answer a few questions about your dataset.')




    if linear_data == True:

        reasons.append('Your dataset is linear.')


    for reason in reasons:
        st.write(reason)
