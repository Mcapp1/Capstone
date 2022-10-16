import pickle
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# TensorFlow & Keras Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from PIL import Image, ImageOps
from skimage import transform

# title
st.title("Classifying Cancer")

# images list
image_1 = 'streamlit/im1.jpg'
image_2 = 'streamlit/im2.jpg'
image_3 = 'streamlit/adeno.jpg'
image_4 = 'streamlit/large.jpg'
image_5 = 'streamlit/squamous.png'
image_6 = 'streamlit/normal.png'

st.write("Welcome to my streamlit app. At the bottom of the page, you can upload an image to be classified. First, I recommended you read about the model at hand and gain a brief insight into lung cancer diagnoses.")
st.write("Prior to the early 2000s, lung cancer was commonly diagnosed by viewing an x-ray of the chest. Though x-rays are still used, follow-up appointments are often scheduled due to the high volume of both false positive and negative screenings. Low-dose CT scans have demonstrated they are stronger at capturing anomalies. Research has shown that, unlike chest x-rays, yearly CT scans can save lives specifically in high-risk patients. For high-risk patients, getting yearly CT scans before symptoms start helps lower the risk of dying from lung cancer.")
st.write(' ')


# Example
st.text('ct scan                                                                      x-ray')
st.image([image_1, image_2],width=350,clamp=False)
st.write(' ')

# About
st.write('About This Model:')
st.write('Using deep learning neural networks, this model trained on thousands of images so it could accurately determine whether a computed tomography scan of the chest displays cancer. This model is primarily trained on Non-Small Cell Lung Cancer scans.')
st.write('Why is this important?')
st.write('Non-Small Cell Lung Cancer makes up eighty percent of lung cancer cases. Non-Small Cell Lung Cancer is comprised of the following three different types of cancer.')

# Adenocarcinoma
col1,mid,col2 = st.columns([.5,.5,1.5])
with col1:
    st.image(image_3,width=280)
with col2:
    st.write('Adenocarcinoma')
# Large Cell Carcinoma
col3,mid,col4 = st.columns([.5,.5,1.5])
with col3:
    st.image(image_4,width=280)
with col4:
    st.write('Large Cell Carcinoma')
# Squamous Cell Carcinoma    
col5,mid,col6 = st.columns([.5,.5,1.5])
with col5:
    st.image(image_5,width=280)
with col6:
    st.write('Squamous Cell Carcinoma')
    
st.write('The examples of Adenocarcinoma and Large Cell Carcinoma cases are obvious and will even clearly appear on an x-ray. Obvious, is often not the case, as you can see in the Squamous Cell Carcinoma example. For reference, here is a picture of a CT scan of the chest without any notable abnormalities.') 
         
st.image(image_6,width=300)
    
# \nAdenocarcinoma:{st.image(image_3)}\
# \nLarge cell carcinoma{st.image(image_4)}\
# \nand Squamous cell carcinoma{st.image(image_5)}\
# \nWork with your doctor to understand exactly which type of lung cancer you may have and what it means for your treatment options.")

st.write("Here, you can upload any image of a CT scan of the chest. It is recommended the image is an axial slice such as the one depicted above. The model will then return whether it believes the image has cancer in it or not. Additionally, the model will return a percentage which is the model's probability of how confident it believes the classification to be.")


with open('ResNet50_model.pkl','rb') as pickle_in:
    model = pickle.load(pickle_in)

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def predict_upload(file):
    # load and transform image
    np_image = Image.open(file)
    np_image = np.array(np_image).astype('float32')
    np_image = transform.resize(np_image, (256, 256, 1))
    np_image = np.expand_dims(np_image, axis=0)
    
    prediction = model.predict(np_image)
    if prediction <=.5:
        st.write(f"The Model Predicts This Image Displays Cancer\
        \nThe Model is {str(99 - np.round(prediction[0][0]*100)) + '%'} confident in it's prediction")
        
    else:
        st.write(f"The Model Predicts This Image Does Not Display Cancer\
        \nThe Model is {str(np.round(prediction[0][0]*100)) + '%'} confident in it's prediction")
        

if file is None:
    st.text("Please upload an image file")
    
else:
    predict_upload(file)
    st.image(file)
    
st.write(' ')
st.write('Can This Model Be Applied In The Real World?')
st.write('Applied, Yes! but...')
st.write('The purpose of this model is to demonstrate the precision and efficiency that deep learning brings to the healthcare field. After training on just a couple thousand images from real-world datasets, this model demonstrates high accuracy and precision when determining if a scan has cancer or not. Having said that, you should never base a diagnosis on what a model says. The real-world purpose is for a physician to use a model such as this as a reference. By eliminating cases where both a doctor and the model is confident the patient does not have cancer, they can focus on cases they believe do have cancer. Doing so could save lives and using CT scans has already proven to be the case!')
    
    
# disclaimer 
st.markdown(""" <style> .font {
font-size:16px ; font-family: 'Helvetica'; color: #F50808;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">DISCLAIMER: This model is for educational purposes only.\
            \nThe information provided should not be used for diagnosing or treating a health problem or disease, and those seeking personal medical advice should consult with a licensed physician. Always seek the advice of your doctor or other qualified health providers regarding a medical condition.</p>', unsafe_allow_html=True)  

st.write('About the creator of this app:')
st.write("Hello, my name is Michael Capparelli, and I am a Data Scientist. I graduated from SUNY Albany with a biology degree and a psychology minor. After spending several years working in healthcare, I decided to change my career trajectory. Though I enjoyed working in a hospital, shadowing doctors, and overseeing the well-being of 10 social homes, I wasn't using my brain the way I wanted to. After discovering my passion for data science and analytics, I enrolled in General Assembly's Data Science Immersive cohort. You are viewing my capstone project for said cohort and while I still have quite a lot to learn, I am happy and eager to learn more. I hope you enjoyed using the app.")
st.write("Github to entire project - https://github.com/Mcapp1/Capstone")    