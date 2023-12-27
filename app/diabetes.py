import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.stylable_container import stylable_container


def load_model():
    with open('model1/model1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model1 = data["model"]
le_gender = data["le_gender"]
le_smoke = data["le_smoke"]

# @st.cache
def load_data():
    df = pd.read_csv("data/diabetes.csv")
    df['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])
    df['gender'] = le_gender.fit_transform(df['gender'])

    return df
df = load_data()

def show_graph(df, option):
    option = option.replace(" ", "_")
    fig = plt.figure(figsize=(9, 7))
    ax = sns.kdeplot(data=df, x=option, hue='diabetes', fill=True)

    if option == "gender":
        label_mapping = {0: 'Female', 1: 'Male', 2: 'Other'}
        # Customize x-axis labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([label_mapping[label] for label in ax.get_xticks()])

        # Remove specific points from the x-axis
        ax.set_xticks([0.5, 1, 1.5], minor=True)
        ax.set_xticklabels([], minor=True)
    
    if option == "smoking_history":
        label_mapping = {4: 'Never', 0: 'No Info',  1: 'Current', 3: ' Former',  2: 'Ever', 5: 'Not Current'}
        # Customize x-axis labels
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels([label_mapping[label] for label in ax.get_xticks()])

        # Remove specific points from the x-axis
        ax.set_xticks([0.5, 1, 1.5], minor=True)
        ax.set_xticklabels([], minor=True)

    st.pyplot(fig)
    
def heatmap(df):
    fig = plt.figure(figsize=(9, 7))

    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    st.pyplot(fig)
    
def op(x):
    return x.replace("_"," ")
 
def add_predictions(diabetes):
  
  st.subheader("Diabetes prediction result:")
  
  
  if diabetes == 0:
    st.write("<span class='diagnosis benign'>Negative</span>,  indicating the absence of diabetes", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Positive</span>, indicating the presence of diabetes", unsafe_allow_html=True)
    
  
  with stylable_container(
            key="description", 
            css_styles=[
    """{
        padding: 0.2rem;
    }
    """
            ]
        ): st.write("While the Diabetes Predictor App provides valuable insights, it is crucial to note that the predictions are based on statistical analysis and machine learning algorithms. The results should not replace professional medical advice. Users are encouraged to consult with healthcare professionals for accurate diagnosis and personalized health guidance.")




def diabetes():
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    st.title("Diabetes Prediction")

    st.write("""Welcome to the Diabetes Predictor App, a powerful tool designed to assist in predicting the likelihood of diabetes based on comprehensive medical and demographic data. This app utilizes a Logistic Regression machine learning model trained on a dataset containing essential features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.""")

    gender = (
        "Female", 
        "Male", 
        "Other"
    )

    smoking_hist = (
        'never', 'No Info', 'current', 'former', 'ever', 'not current'
    )
    
    st.sidebar.header("Patient Information")
    gender = st.sidebar.selectbox("Gender", gender)
    smorking_history = st.sidebar.selectbox("Smoking History", smoking_hist)

    age = st.sidebar.slider("Age", 0, 80, 1)
    hypertension = st.sidebar.selectbox("Hypertension", [True, False])
    heart_disease = st.sidebar.selectbox("Heart Disease", [True, False])
    bmi = st.sidebar.slider("BMI", 0.0, df.bmi.max(), 0.01)
    blood_glucose_level = st.sidebar.slider("Blood glucose level", 0, df.blood_glucose_level.max(), 1)
    HbA1c_level = st.sidebar.slider("HbA1c level", 0.0, df.HbA1c_level.max(), 0.1)
    if hypertension:
        hypertension = 1
    else:
        hypertension = 0
    if heart_disease:
        heart_disease=1
    else:
        heart_disease = 0
    X = np.array([[gender, age,hypertension,heart_disease, smorking_history, bmi,HbA1c_level, blood_glucose_level  ]])
    X[:, 0] = le_gender.transform(X[:,0])
    X[:, 4] = le_smoke.transform(X[:,4])
    X = X.astype(float)

    diabetes = model1.predict(X)
        

    
    col = df.columns
    col=col[:(len(col)-1)]
    col = col.map(op)

        
    col1, col2 = st.columns([2, 1])
    with col1:  
        
        st.write("##### Choose option to display the diabete graph below")
        option = st.selectbox("Options", col )  
        st.write(f"## Distribution of Diabetes in {option}")  
        show_graph(df, option)
                
        st.write("## Correlation Matrix")

        heatmap(df)
    
    with col2:
        with stylable_container(
            key="predict", 
            css_styles=[
    """{
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #7E99AB;
    }
    """
            ]
        ):
            add_predictions(diabetes)