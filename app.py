import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
import streamlit as st
import joblib

df = pd.read_csv(r"C:\Users\Samarth\Downloads\heart_failure_clinical_records_dataset.csv")
df

x=df.drop(["age"],axis=1)
y=df["smoking"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

pickle_out = open("model.pkl","wb")
pickle.dump(knn, pickle_out)
pickle_out.close()

st.markdown('## Heart Failure Prediction')
age = st.number_input('age')
high_blood_pressure = st.number_input('high_blood_pressure')
smoking = st.number_input('smoking')
diabetes = st.number_input('diabetes')

if st.button('Predict'):
    model = joblib.load('model.pkl')
    x = np.array([age, high_blood_pressure, smoking, diabetes])
    if any(x <= 0):
        st.markdown('## Inputs must be greater than 0')
    else:
        st.markdown(f'## Prediction is {model.predict([[age, high_blood_pressure, smoking, diabetes]])}')

