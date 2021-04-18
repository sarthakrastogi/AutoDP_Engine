import streamlit as st

def app():
    st.write('The app collects anonymous statistics on your dataset so as to predict better weights for future datasets we receive. It performs online learning to update the weights on the ML models we have with us to improve the app. We do not store any sensitive information.')
    st.header('What we store:')
    st.write('* The shape of your dataset

            * The skewness of your features

            * The ')
