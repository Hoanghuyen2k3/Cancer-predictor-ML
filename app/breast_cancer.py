import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_extras.stylable_container import stylable_container

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 0'], axis=1)
    data['y'] = data['y'].map({'M':1, 'B':0})
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")
    data = get_clean_data()
    print(data.info())
    slider_labels = [
        ("Radius (mean)", "x.radius_mean"),
        ("Texture (mean)", "x.texture_mean"),
        ("Perimeter (mean)", "x.perimeter_mean"),
        ("Area (mean)", "x.area_mean"),
        ("Smoothness (mean)", "x.smoothness_mean"),
        ("Compactness (mean)", "x.compactness_mean"),
        ("Concavity (mean)", "x.concavity_mean"),
        ("Concave points (mean)", "x.concave_pts_mean"),
        ("Symmetry (mean)", "x.symmetry_mean"),
        ("Fractal dimension (mean)", "x.fractal_dim_mean"),
        ("Radius (se)", "x.radius_se"),
        ("Texture (se)", "x.texture_se"),
        ("Perimeter (se)", "x.perimeter_se"),
        ("Area (se)", "x.area_se"),
        ("Smoothness (se)", "x.smoothness_se"),
        ("Compactness (se)", "x.compactness_se"),
        ("Concavity (se)", "x.concavity_se"),
        ("Concave points (se)", "x.concave_pts_se"),
        ("Symmetry (se)", "x.symmetry_se"),
        ("Fractal dimension (se)", "x.fractal_dim_se"),
        ("Radius (worst)", "x.radius_worst"),
        ("Texture (worst)", "x.texture_worst"),
        ("Perimeter (worst)", "x.perimeter_worst"),
        ("Area (worst)", "x.area_worst"),
        ("Smoothness (worst)", "x.smoothness_worst"),
        ("Compactness (worst)", "x.compactness_worst"),
        ("Concavity (worst)", "x.concavity_worst"),
        ("Concave points (worst)", "x.concave_pts_worst"),
        ("Symmetry (worst)", "x.symmetry_worst"),
        ("Fractal dimension (worst)", "x.fractal_dim_worst"),
    ]
    
    input = {}
    for label, key in slider_labels:
        input[key]= st.sidebar.slider(
            label, 
            min_value=float(0), 
            max_value=float(data[key].max()),
            value= float(data[key].mean())
        )
    return input

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['y'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  with stylable_container(
            key="description", 
            css_styles=[
    """{
        padding: 0.2rem;
    }
    """
            ]
        ): st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis")



def get_radar_chart(input):
    input_data = get_scaled_values(input)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['x.radius_mean'], input_data['x.texture_mean'], input_data['x.perimeter_mean'],
            input_data['x.area_mean'], input_data['x.smoothness_mean'], input_data['x.compactness_mean'],
            input_data['x.concavity_mean'], input_data['x.concave_pts_mean'], input_data['x.symmetry_mean'],
            input_data['x.fractal_dim_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['x.radius_se'], input_data['x.texture_se'], input_data['x.perimeter_se'], input_data['x.area_se'],
            input_data['x.smoothness_se'], input_data['x.compactness_se'], input_data['x.concavity_se'],
            input_data['x.concave_pts_se'], input_data['x.symmetry_se'],input_data['x.fractal_dim_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['x.radius_worst'], input_data['x.texture_worst'], input_data['x.perimeter_worst'],
            input_data['x.area_worst'], input_data['x.smoothness_worst'], input_data['x.compactness_worst'],
            input_data['x.concavity_worst'], input_data['x.concave_pts_worst'], input_data['x.symmetry_worst'],
            input_data['x.fractal_dim_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig


def show_breast_cancer():
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input = add_sidebar()
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Link this application to your cytology laboratory for precise breast cancer diagnostics from your tissue samples. This app utilizes a sophisticated machine learning model to forecast whether a breast mass is benign or malignant, relying on the measurements obtained from your cytology lab. Alternatively, you have the option to manually input and update the measurements through the user-friendly sliders located in the sidebar.")
          
    col1, col2 = st.columns([3, 1])
    with col1:
        chart = get_radar_chart(input)
        st.plotly_chart(chart)
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
            add_predictions(input)
        
    


# if __name__ == "__main__":
#     main()