import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model('vgg16_model.h5')

disease_info = {
    'Pepper__bell___Bacterial_spot': 'Bacterial spot is a common disease of pepper plants. It is caused by the bacterium Xanthomonas campestris pv. vesicatoria and can lead to reduced yield and quality of the fruit.',
    'Pepper__bell___healthy': 'Pepper plants that are healthy and free from disease have dark green leaves and strong stems. They are able to produce a high yield of fruit.',
    'Potato___Early_blight': 'Early blight is a fungal disease that affects potato plants. It causes brown spots on the leaves and stems, and can lead to reduced yield and quality of the tubers.',
    # Add information about other diseases here
}


# Create a list of class names
class_names = ['Pepper__bell___Bacterial_spot',  # 0
               'Pepper__bell___healthy',  # 1
               'Potato___Early_blight',  # 2
               'Potato___Late_blight',  # 3
               'Potato___healthy',  # 4
               'Tomato_Bacterial_spot',  # 5
               'Tomato_Early_blight',  # 6
               'Tomato_Late_blight',  # 7
               'Tomato_Leaf_Mold',  # 8
               'Tomato_Septoria_leaf_spot',  # 9
               'Tomato_Spider_mites_Two_spotted_spider_mite',  # 10
               'Tomato__Target_Spot',  # 11
               'Tomato__Tomato_YellowLeaf__Curl_Virus',  # 12
               'Tomato__Tomato_mosaic_virus',  # 13
               'Tomato_healthy',  # 14
               ]

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to make predictions on the uploaded image
def predict_disease(image):
    img = preprocess_image(image)
    pred = model.predict(img)
    class_index = np.argmax(pred[0])
    st.write("Index : "+str(class_index))
    class_name = class_names[class_index]
    confidence_score = round(np.max(pred[0]) * 100, 2)
    return class_name, confidence_score

# Create the web app
st.set_page_config(page_title='Plant Disease Prediction', page_icon=':seedling:')
st.title('Plant Disease Prediction')

# Upload image
uploaded_file = st.file_uploader(
    'Upload an image of a plant', type=['jpg', 'jpeg', 'png'])

# Make prediction
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded plant image', use_column_width=True)

    # Make prediction on uploaded image
    image_array = np.array(image)
    class_name, confidence_score = predict_disease(image_array)

    # Display predicted class label and confidence score
    st.write(f'Predicted class label: {class_name}')
    st.write(f'Confidence score: {confidence_score}%')

    # Display information about the disease
    if class_name in disease_info:
        st.write(disease_info[class_name])
