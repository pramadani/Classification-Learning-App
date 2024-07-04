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

dashboard = st.Page(
    "dashboard.py", 
    title="Recognition App", 
    icon=":material/videocam:", 
    default=True
)

predict = st.Page(
    "prediksi.py", 
    title="Image Recognition", 
    icon=":material/image:"
)

pg = st.navigation(
    {
        
        "Dashboard": [dashboard, predict],
    }
)

pg.run()