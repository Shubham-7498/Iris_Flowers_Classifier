import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


model = joblib.load('iris_classifier.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
)

with st.container():
    st.title("🌸 Iris Flower Classifier")

    st.header("Enter the Flower Measurements:")

    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.0)
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 1.3)

    if st.button("Predict Species"):
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.success(f"🌼 Predicted Iris Species: **{prediction[0].capitalize()}**")


st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
