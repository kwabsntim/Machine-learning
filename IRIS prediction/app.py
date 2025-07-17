#Importing all dependecies 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 


#loading the csv file into pandas
iris=pd.read_csv("Iris.csv") 

#Data preprocessing 
X=iris.drop(["Id","Species"],axis=1)
y=iris["Species"]

#Spliting the dataset into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Model training 

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

st.title("IRIS FLOWER PREDICTION APP 🌷")

#Getting input from the user 

Sepal_width=st.number_input("Enter your sepal_width(Cm)",min_value=0.0,max_value=10.0,step=0.1,key="sepal_width")
Sepal_length=st.number_input("Enter your sepal_length(Cm)",min_value=0.0,max_value=10.0,step=0.1,key="sepal_length")
Petal_width=st.number_input("Enter your petal_width(Cm)",min_value=0.0,max_value=10.0,step=0.1,key="petal_width")
Petal_length=st.number_input("Enter your petal_length(Cm)",min_value=0.0,max_value=10.0,step=0.1,key="petal_length")

#Model prediction
input_data=[[Sepal_length,Sepal_width,Petal_length,Petal_width]]
prediction=model.predict(input_data)

#Displaying the predicted value 
if st.button("Predict"):
    st.success(f'The predicted value is {prediction}')