import pickle
import streamlit as st

st.title('Classifying CT Scans For Whether The Scan Displays Cancer')
st.write('Upload an Image')

with open('ResNet50_model.pkl','rb') as pickle_in:
    model = pickle.load(pickle_in)
    
text = st.text_input('Image Here:'value=input)

predicted_scan = 