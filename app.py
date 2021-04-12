import streamlit as st
from multiapp import MultiApp
from apps import home, split_features, ensure_numerical, ensure_categorical, outliers, missing_values, target_feature, encoding, train_test_split_data, recommend_algorithm, select_model, predict
app = MultiApp()

st.title('The Data Science Application')
st.header('Sarthak Rastogi | thesarthakrastogi@gmail.com')

# Add all your application here
app.add_app("Home", home.app) #1
app.add_app("Dividing numerical and categorical features", split_features.app)#2
app.add_app('Ensure numerical', ensure_numerical.app) #2A
app.add_app("Ensure categorical", ensure_categorical.app) #2B
app.add_app("Handling outliers", outliers.app)#3
app.add_app("Handling missing values", missing_values.app)#4
app.add_app("Picking a target feature", target_feature.app)#5
app.add_app("Encoding Values", encoding.app)#6
app.add_app("Splitting data into train and test sets", train_test_split_data.app)#7
app.add_app("Recommend algorithm", recommend_algorithm.app)#8
app.add_app("Picking an algorithm", select_model.app)#9
app.add_app("Making Predictions", predict.app)#10
# The main app
app.run()
