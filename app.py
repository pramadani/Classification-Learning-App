import streamlit as st

logo = st.logo("resources/logo_crop.png")

dashboard = st.Page(
    "dashboard.py", 
    title="Dashboard", 
    icon=":material/dashboard:", 
    default=True
)
data = st.Page(
    "Tutorial/data.py", 
    title="Dataset", 
    icon=":material/database:"
)
preparation = st.Page(
    "Tutorial/preparation.py", 
    title="Data Preparation", 
    icon=":material/data_thresholding:"
)
training = st.Page(
    "Tutorial/training.py", 
    title="Training", 
    icon=":material/model_training:"
)
testing = st.Page(
    "Tutorial/testing.py", 
    title="Testing", 
    icon=":material/experiment:"
)
predict = st.Page(
    "Predict/prediction-app.py", 
    title="Iris App", 
    icon=":material/output:"
)

pg = st.navigation(
    {
        
        "Dashboard": [dashboard],
        "Predict": [predict],
        "Explanation": [data, preparation, training, testing],
    }
)

pg.run()