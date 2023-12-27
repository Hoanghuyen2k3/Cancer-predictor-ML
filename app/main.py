import streamlit as st
from breast_cancer import show_breast_cancer
from diabetes import diabetes

def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
        
        
    )

    page = st.sidebar.selectbox("Predictor App", ("Breast Cancer Predictor", "Diabetes Predictor"))

    if page == "Breast Cancer Predictor":
        show_breast_cancer()
    else:
        diabetes()
if __name__ == "__main__":
    main()