import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px


st.title("Transpoprt")

#rasm yuklash
file = st.file_uploader("Rasm yuklash", type=['jpeg','png','gif','svg','jpg'])

if file:
    st.image(file)
    #PIL CONVERT
    img = PILImage.create(file)
    model = load_learner('Transorttt_model.pkl')
    
    #predection
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimolligi: {probs[pred_id]*100:.1f}%")

    #lotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)