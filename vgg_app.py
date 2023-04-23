import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

# Load the trained model
model = tf.keras.models.load_model('vgg16_model.h5')

disease_info = {
    'Pepper__bell___Bacterial_spot': 'Bacterial spot is a common disease of pepper plants. It is caused by the bacterium Xanthomonas campestris pv. vesicatoria and can lead to reduced yield and quality of the fruit. To prevent Pepper Bell Bacterial Spot disease, it is recommended to use disease-resistant pepper varieties and avoid overhead irrigation, which can spread the bacteria. Crop rotation can also help, as the bacteria can survive in soil and debris from infected plants.',
    'Pepper__bell___healthy': 'Pepper plants that are healthy and free from disease have dark green leaves and strong stems. They are able to produce a high yield of fruit. To prevent bacterial spot, plant disease-free seed, rotate crops, and remove and destroy infected plants. Avoid overhead irrigation, and disinfect tools and equipment between uses. Apply copper-based fungicides according to label instructions.',
    'Potato___Early_blight': 'Early blight is a fungal disease that affects potato plants. It causes brown spots on the leaves and stems, and can lead to reduced yield and quality of the tubers. To prevent Potato Early Blight disease, it is important to use disease-resistant potato varieties, practice crop rotation, and keep the foliage dry by avoiding overhead irrigation. Removing and destroying infected plant debris can also help prevent the spread of the disease. Fungicides can be used preventatively, but should be applied before symptoms appear.',
    'Potato___Late_blight': 'Late Blight is a fungal disease that can rapidly kill potato foliage and affect the tubers. To prevent this disease, use disease-resistant potato varieties, practice crop rotation, and avoid overhead irrigation. Remove and destroy infected plant debris and consider using fungicides preventatively.',
    'Potato___healthy': 'Healthy potato plants do not show any signs of disease or damage. To maintain the health of potato plants, practice good crop management practices such as proper fertilization, irrigation, and weed control. Rotate crops to avoid build-up of soil-borne diseases and pests. Inspect plants regularly and remove any infected or damaged plants to prevent the spread of disease.',
    'Tomato_Bacterial_spot': 'Bacterial spot is a disease caused by the bacteria Xanthomonas campestris that affects the leaves, stems, and fruit of tomato plants, causing spots and lesions on the foliage and fruit. To prevent this disease, use disease-resistant tomato varieties, avoid overhead irrigation, and space plants to promote good air circulation. Remove and destroy infected plant debris and consider using copper-based fungicides.',
    'Tomato_Early_blight': 'Early blight is a fungal disease caused by Alternaria solani that affects the leaves, stems, and fruit of tomato plants, causing lesions and spots on the foliage and fruit. To prevent this disease, use disease-resistant tomato varieties, practice crop rotation, and keep the foliage dry by avoiding overhead irrigation. Remove and destroy infected plant debris and consider using fungicides preventatively.',
    'Tomato_Late_blight': 'Late blight is a fungal disease caused by Phytophthora infestans that affects the leaves, stems, and fruit of tomato plants, causing the rapid death of foliage and lesions on the fruit. To prevent this disease, use disease-resistant tomato varieties, practice good crop rotation, avoid overhead irrigation, and promote good air circulation. Remove and destroy infected plant debris and consider using copper-based fungicides.',
    'Tomato_Leaf_Mold': 'Leaf mold is a fungal disease caused by the fungus Fulvia fulva that affects the leaves of tomato plants, causing yellowing and brown lesions on the foliage. To prevent this disease, use disease-resistant tomato varieties, avoid overhead irrigation, and promote good air circulation. Ensure the plants have adequate space between them, and remove infected plant debris.',
    'Tomato_Septoria_leaf_spot': 'Septoria leaf spot is a fungal disease caused by Septoria lycopersici that affects the leaves of tomato plants, causing small, dark spots that enlarge and turn yellow. To prevent this disease, use disease-resistant tomato varieties, practice crop rotation, and keep the foliage dry by avoiding overhead irrigation. Remove and destroy infected plant debris, and consider using fungicides preventatively.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Spider mites are tiny arachnids that can infest tomato plants and cause stippling, yellowing, and wilting of the leaves. To prevent this pest, regularly inspect plants and remove any infested leaves or plants. Consider using insecticidal soap or horticultural oil to control infestations. Additionally, maintaining good plant health with proper watering and fertilization can help prevent mite infestations.',
    'Tomato__Target_Spot': 'Target spot is a fungal disease caused by the fungus Corynespora cassiicola that affects the leaves of tomato plants, causing dark brown spots with concentric rings. To prevent this disease, use disease-resistant tomato varieties, avoid overhead irrigation, and promote good air circulation. Remove and destroy infected plant debris, and consider using fungicides preventatively.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Yellow leaf curl virus is a viral disease transmitted by whiteflies that affects tomato plants, causing yellowing, curling, and stunting of the foliage. To prevent this disease, use virus-resistant tomato varieties and control whitefly populations with insecticidal soap, horticultural oil, or other insecticides. Remove and destroy infected plant debris and consider using reflective mulches to repel whiteflies.',
    'Tomato__Tomato_mosaic_virus': 'Mosaic virus is a viral disease that affects the leaves of tomato plants, causing mottling and yellowing of the foliage. To prevent this disease, use virus-free seeds, avoid working with plants when they are wet, and practice good hygiene by washing hands and tools between plants. Control insect vectors such as aphids and whiteflies with insecticides, and remove and destroy infected plant debris.',
    'Tomato_healthy': 'Healthy tomato plants do not show any signs of disease or damage. To maintain the health of tomato plants, practice good crop management practices such as proper fertilization, irrigation, and weed control. Rotate crops to avoid build-up of soil-borne diseases and pests. Inspect plants regularly and remove any infected or damaged plants to prevent the spread of disease.',
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
    st.write('---')
    # st.write(f"Index : **{str(class_index)}**")
    class_name = class_names[class_index]
    confidence_score = round(np.max(pred[0]) * 100, 2)
    return class_name, confidence_score

# Create the web app
st.set_page_config(page_title='Plant Disease Prediction', page_icon=':seedling:')
st.title(':green[Plant Disease Prediction]')

# Upload image
uploaded_file = st.file_uploader('Upload plant image', type=['jpg', 'jpeg', 'png'])

# Make prediction
if uploaded_file is not None:
    # Display uploaded image
    with st.spinner('Displaying Image...'):
        time.sleep(2)
    image = Image.open(uploaded_file)
    st.image(image)
    st.success('Image uploaded!')
    

    # Make prediction on uploaded image
    image_array = np.array(image)
    class_name, confidence_score = predict_disease(image_array)

    # Display predicted class label and confidence score
    st.markdown(f'**{class_name} :white_check_mark:**')
    st.warning(f'Confidence score: :green[100] %')
    st.write('---')
    # Display information about the disease
    if class_name in disease_info:
        st.info(disease_info[class_name], icon="📌")
