import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# load data
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

st.title("Iris Flower Prediction")
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 5.5)
petal_width = st.slider("Petal Width", 2.0, 7.0, 5.5)

prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write(f"Predicted class: {iris.target_names[prediction[0]]}")