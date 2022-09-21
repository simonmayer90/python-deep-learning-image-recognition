# imports
import streamlit as st
import pandas as pd
import numpy as np

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) 

install('cv2')

import cv2

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
### Please upload the image of the template PCB!
"""
)

# file uploader
uploaded_template_file = st.file_uploader("Choose a file", key=1)

st.markdown("""
### Please upload the image of the test PCB!
"""
)

# file uploader
uploaded_test_file = st.file_uploader("Choose a file", key=2)

inspectButton = st.button('Inspect!')

if inspectButton == 1:

    img1 = cv2.imread(uploaded_template_file,cv2.IMREAD_COLOR)  #reading the images
    img2 = cv2.imread(uploaded_test_file,cv2.IMREAD_COLOR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)        #converting to gray scale
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray1,200,255,cv2.THRESH_BINARY)  #applying threshold (binary) 
    ret,thresh2 = cv2.threshold(gray2,200,255,cv2.THRESH_BINARY)
    res =cv2.bitwise_xor(thresh1, thresh2, mask=None)      #comparing the images     
    cv2.imshow(res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    st.write("Inspecting...")
    st.stop()
