# imports
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package]) 

# install('opencv-python')
# install('imutils')

import cv2
import imutils

st.set_page_config(page_title="PCB Inspector",
                   page_icon="🔍",
                   # initial_sidebar_state="expanded",
                   # layout="wide"
                   )

st.markdown("<h1 style='text-align: center;'>🔍 PCB Inspector 🔍</h1>", unsafe_allow_html=True)

# st.image("https://github.com/simonmayer90/python-deep-learning-image-recognition/raw/4d89104da41a027d2d6730f70761eb7e23e37fe1/pcb-image.jpg")

st.markdown("""
### Please upload the image of the template PCB:
"""
)

uploaded_template_img = st.file_uploader("Choose a file", key=1)

st.markdown("""
### Please upload the image of the test PCB:
"""
)

uploaded_test_img = st.file_uploader("Choose a file", key=2)

transformNeeded = st.checkbox('The test PCB is rotated / has another image size.')

inspectButton = st.button('Inspect!')

cnn = keras.models.load_model("cnn_model2.h5", compile=False)


if inspectButton == 1:
    template_img = Image.open(uploaded_template_img)
    test_img = Image.open(uploaded_test_img)

    converted_template_img = np.array(template_img.convert('RGB'))
    converted_test_img = np.array(test_img.convert('RGB'))

    cv2.imwrite("template.jpg", converted_template_img)
    cv2.imwrite("test.jpg", converted_test_img)

    rgb_template_img = cv2.imread('template.jpg')
    rgb_test_img = cv2.imread('test.jpg')  

    # read images as grayscale
    gray_template_img = cv2.cvtColor(rgb_template_img, cv2.COLOR_BGR2GRAY)
    gray_test_img = cv2.cvtColor(rgb_test_img, cv2.COLOR_BGR2GRAY)

    # Rotation

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_test_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_template_img, None)

    # Match features. Matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # sort matches
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(rgb_test_img, keypoints1, rgb_template_img, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.LMEDS)

    # Use homography
    height, width, channels = rgb_template_img.shape
    rgb_test_img_transf = cv2.warpPerspective(rgb_test_img, h, (width, height))

    # cut border
    width = rgb_template_img.shape[1]
    height = rgb_template_img.shape[0]

    rgb_template_img = rgb_template_img[10:height-10, 10:width-10]
    rgb_test_img_transf = rgb_test_img_transf[10:height-10, 10:width-10]

    rgb_test_img_transf_show = rgb_test_img_transf.copy()

    # read images as grayscale
    gray_template_img = cv2.cvtColor(rgb_template_img, cv2.COLOR_BGR2GRAY)
    gray_test_img_transf = cv2.cvtColor(rgb_test_img_transf, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to blur the image before thresholding
    blur_template_img = cv2.GaussianBlur(gray_template_img, (3,3),0)
    blur_test_img = cv2.GaussianBlur(gray_test_img_transf, (3,3),0)

    # Adaptive thresholding(mean)
    # Thresholding is used to turn a grayscale image into a binary image based on a 
    # specific threshold value
    template_adap_thresh = cv2.adaptiveThreshold(blur_template_img, 255, 
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    test_adap_thresh = cv2.adaptiveThreshold(blur_test_img, 255, 
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Image subtraction (template - test)
    sub_img= cv2.subtract(template_adap_thresh, test_adap_thresh)

    # Median blur to eliminate background noise
    blur_img = cv2.medianBlur(sub_img, 9)
    
    # diff = cv2.absdiff(template_adap_thresh, test_adap_thresh)

    # diff = cv2.absdiff(template_img_resize, test_img)

    # thresh = cv2.threshold(final_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # dilation to make defected areas bigger
    kernel = np.ones((5,5), np.uint8)
    dilate_img = cv2.dilate(blur_img, kernel, iterations=14)
    
    contours = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(rgb_template_img, (x,y), (x+w, y+h), (255,0,0), 3)
            cv2.rectangle(rgb_test_img_transf, (x,y), (x+w, y+h), (255,0,0), 3)

            crop_defect = rgb_test_img_transf[y:y+h, x:x+w]    
            cv2.imwrite("crop_defect.jpg", crop_defect)

            test_image = image.load_img("crop_defect.jpg", target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = cnn.predict(test_image)

            if np.argmax(result) == 0:
                prediction = 'Missing_hole'
            elif np.argmax(result) == 1:
                prediction = 'Mouse_bite'
            elif np.argmax(result) == 2:
                prediction = 'Open_circuit'
            elif np.argmax(result) == 3:
                prediction = 'Short'
            elif np.argmax(result) == 4:
                prediction = 'Spur'
            elif np.argmax(result) == 5:
                prediction = 'Spurious_copper'
            else:
                prediction = 'Unknown_defect'

            cv2.putText(rgb_test_img_transf, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)


    # x = np.zeros((360, 10, 3), np.uint8)
    # result = np.hstack((rgb_template_img, x, rgb_test_img))

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

    st.markdown("""
    ## Matches between Template and Test PCB:
    """
    )

    st.image(imMatches)

    st.markdown("""
    ## Transformed Template PCB:
    """
    )

    st.image(blur_template_img)
    st.image(template_adap_thresh)


    st.markdown("""
    ## Transformed Test PCB:
    """
    )

    st.image(rgb_test_img_transf_show)
    st.image(blur_test_img)
    st.image(test_adap_thresh)

    st.markdown("""
    ## Substraction of Images:
    """
    )

    st.image(sub_img)
    st.image(blur_img)
    st.image(dilate_img)
    
    st.markdown("""
    ## Areas of defects:
    """
    )

    st.image(rgb_template_img)
    st.image(rgb_test_img_transf)

    # contour detection to get the count of defects 
    cnts = cv2.findContours(dilate_img, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]          
    blobs = []
    for cnt in cnts:
        if 0<cv2.contourArea(cnt)<100000:
            blobs.append(cnt)  

    st.text("Number of defects in Test PCB:")
    st.text(len(blobs))



