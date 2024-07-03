import streamlit as st

st.set_page_config(page_title="Classification App",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## Developed by  \n" + "Ararya Pramadani Alief Rahman  \n" + "Azarya Santoso  \n" + "Made Diksa Pitra  \n"
    })

st.logo("resources/logo_large.png", icon_image="resources/logo_crop.png")

st.image("resources/logo_crop.png", width=150)

st.write("\n")
st.write("\n")
st.write("\n")

dashboard = st.Page(
    "dashboard.py", 
    title="Dashboard", 
    icon=":material/dashboard:", 
    default=True
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

predict2 = st.Page(
    "prediksi.py", 
    title="Prediksi", 
    icon=":material/output:"
)

pg = st.navigation(
    {
        
        "Dashboard": [dashboard],
        "Predict": [predict, predict2],
        "Explanation": [preparation, training, testing],
    }
)

pg.run()