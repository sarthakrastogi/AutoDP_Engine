import streamlit as st
from multiapp import MultiApp
from apps import a_home, b_split_features, c_ensure_numerical, d_ensure_categorical, e_outliers, f_missing_values, g_target_feature, h_encoding, i_train_test_split_data, j_recommend_algorithm, k_select_model, l_predict
app = MultiApp()

st.title('Data Science One')
st.header('An End-to-End Data Science Application by')
st.header('Sarthak Rastogi | thesarthakrastogi@gmail.com')

# Add all your application here
app.add_app("Home", a_home.app) #1
app.add_app("Numerical vs categorical features", b_split_features.app)#2
app.add_app('Ensure numerical', c_ensure_numerical.app) #2A
app.add_app("Ensure categorical", d_ensure_categorical.app) #2B
app.add_app("Handling outliers", e_outliers.app)#3
app.add_app("Handling missing values", f_missing_values.app)#4
app.add_app("Picking a target feature", g_target_feature.app)#5
app.add_app("Encoding Values", h_encoding.app)#6
app.add_app("Splitting data into train and test sets", i_train_test_split_data.app)#7
app.add_app("Recommend algorithm", j_recommend_algorithm.app)#8
app.add_app("Picking an algorithm", k_select_model.app)#9
app.add_app("Making Predictions", l_predict.app)#10
# The main app
app.run()
