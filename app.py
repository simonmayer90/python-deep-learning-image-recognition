# imports
import streamlit as st
import pandas as pd
import numpy as np
# from PIL import Image

# %% STREAMLIT
# Set configuration
st.set_page_config(page_title="PCB Inspector",
                   page_icon="🔍",
                   initial_sidebar_state="expanded",
                   # layout="wide"
                   )

# set colors: These has to be set on the setting menu online
    # primary color: #FF4B4B, background color:#0E1117
    # text color: #FAFAFA, secindary background color: #E50914

# Set the logo of app
# st.sidebar.image("pcb_inspector_logo.png", width=300, clamp=True)
# welcome_img = Image.open('welcome_page_img.png')
# st.image(welcome_img)
st.markdown("<h1 style='text-align: center;'>🔍 PCB Inspector 🔍</h1>", unsafe_allow_html=True)

# image = Image.open('https://github.com/simonmayer90/python-deep-learning-image-recognition/blob/4d89104da41a027d2d6730f70761eb7e23e37fe1/pcb-image.jpg')

st.image("https://github.com/simonmayer90/python-deep-learning-image-recognition/raw/4d89104da41a027d2d6730f70761eb7e23e37fe1/pcb-image.jpg")

# %% APP WORKFLOW
st.markdown("""
### Please upload the image of the printed circuit board that you want to inspect!
"""
)

# file uploader
uploaded_file = st.file_uploader("Choose a file")

inspectButton = st.button('Inspect!')

if inspectButton == 1:
    st.write("Inspecting...")
