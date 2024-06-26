import streamlit as st

dashboard = st.Page(
    "dashboard.py", 
    title="Dashboard", 
    icon=":material/dashboard:", 
    default=True
)
data = st.Page(
    "Tutorial/data.py", 
    title="Dataset", 
    # icon=":material/radioactive_sign:"
)
preparation = st.Page(
    "Tutorial/preparation.py", 
    title="Data Preparation", 
    # icon=":material/notification_important:"
)
training = st.Page(
    "Tutorial/training.py", 
    title="Training", 
    # icon=":material/notification_important:"
)
testing = st.Page(
    "Tutorial/testing.py", 
    title="Testing", 
    # icon=":material/notification_important:"
)

pg = st.navigation(
    {
        "Dashboard": [dashboard],
        "Explanation": [data, preparation, training, testing],
    }
)

# pg = st.navigation([login_page])

pg.run()