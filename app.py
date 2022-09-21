# imports
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from PIL import Image

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) 

install('opencv-python')

import cv2

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
uploaded_template_img = st.file_uploader("Choose a file", key=1)

st.markdown("""
### Please upload the image of the test PCB!
"""
)

# file uploader
uploaded_test_img = st.file_uploader("Choose a file", key=2)

inspectButton = st.button('Inspect!')

if inspectButton == 1:
    template_img = Image.open(uploaded_template_img)
    test_img = Image.open(uploaded_test_img)

    converted_template_img = np.array(template_img.convert('RGB'))
    converted_test_img = np.array(test_img.convert('RGB'))

    st.markdown("""
    ## Template PCB:
    """
    )
    st.image(converted_template_img)

    st.markdown("""
    ## Test PCB:
    """
    )
    st.image(converted_test_img)


    # gray1 = cv2.cvtColor(converted_template_img, cv2.COLOR_BGR2GRAY)        #converting to gray scale
    # gray2 = cv2.cvtColor(converted_test_img, cv2.COLOR_BGR2GRAY)
    # ret,thresh1 = cv2.threshold(gray1,200,255,cv2.THRESH_BINARY)  #applying threshold (binary) 
    # ret,thresh2 = cv2.threshold(gray2,200,255,cv2.THRESH_BINARY)
    # res =cv2.bitwise_xor(thresh1, thresh2, mask=None)      #comparing the images     
    # st.image(res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rgb_template_img = cv2.imread(np.array(template_img.convert('RGB')))  

    # read template PCB 01 image as grayscale image
    template_img = cv2.imread(rgb_template_img, 0)  
    # the 2nd parameter is flag, makes image grayscale for value 0 or 2

    # resize template image of PCB
    template_img_resize = cv2.resize(template_img, (750, 450))

    # Gaussian blur to blur the image before thresholding
    blur_template_img = cv2.GaussianBlur(template_img_resize, (3,3),0)


    # Adaptive thresholding(mean)
    # Thresholding is used to turn a grayscale image into a binary image based on a 
    # specific threshold value
    template_adap_thresh = cv2.adaptiveThreshold(blur_template_img, 255, 
                                            cv2. ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15, 5)


    # read test image of PCB 01
    rgb_test_img = cv2.imread(np.array(test_img.convert('RGB')))  

    # read grayscale test PCB image
    test_img = cv2.imread(rgb_test_img, 0)

    # resize test image of PCB
    test_img_resize = cv2.resize(test_img, (750, 450))

    # Gaussian blur to blur the image before thresholding
    blur_test_img = cv2.GaussianBlur(test_img_resize, (3,3),0)

    # Adaptive thresholding(mean) on test image
    test_adap_thresh = cv2.adaptiveThreshold(blur_test_img, 255, 
                                            cv2. ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15, 5)

    # Image subtraction (template - test)

    sub_img= cv2.subtract(template_adap_thresh, test_adap_thresh)

    # Median blur to eliminate background noise
    final_img = cv2.medianBlur(sub_img, 5)

    # display final binary image result 
    # to show defects in the image
    
    st.markdown("""
    ## Areas of defects in the Test PCB:
    """
    )

    st.image(final_img)

    # contour detection to get the count of defects 
    cnts = cv2.findContours(final_img, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]          
    blobs = []
    for cnt in cnts:
        if 0<cv2.contourArea(cnt)<300:
            blobs.append(cnt)  

    st.text("Number of defects in Test PCB:")
    st.text(len(blobs))



