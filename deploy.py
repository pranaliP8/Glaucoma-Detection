import streamlit as st
import tensorflow as tf
from PIL import Image

  
# Load the trained glaucoma detection model
model = tf.keras.models.load_model('c:/Users/vaish/Desktop/MegaProject/results/new_model.h5')

# Define a function to predict the glaucoma risk of an image
def predict_glaucoma_risk(image):
# Preprocess the image
  image = image.convert('RGB')
  image = image.resize((200, 200))
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.expand_dims(image, axis=0)
  
  prediction = model.predict(image)

# Return the predicted class
  if prediction[0][0] > 0.5:
    return 'glaucoma'
  else:
    return 'normal'


# Set the Streamlit page title and header
st.set_page_config(page_title='Glaucoma Detection')
#st.set_page_config(page_bg_color='#FF0000')
st.title('GLAUCOMA DETECTION')
st.header('Upload an image to predict glaucoma risk')

# Add an input area for the user to upload an image
image_file = st.file_uploader('Upload image', type=['jpg', 'png', 'jpeg'])

# If an image is uploaded, predict the glaucoma risk and display the result
if image_file is not None:
  image = Image.open(image_file)
  prediction = predict_glaucoma_risk(image)

  st.write('Prediction:', prediction)

st.sidebar.title('About Glaucoma')
st.sidebar.write('Glaucoma is a group of eye diseases that damage the optic nerve. The optic nerve carries visual information from the eye to the brain. Glaucoma can cause blindness if it is not treated.')
st.sidebar.button('Learn more about glaucoma', on_click=lambda: st.sidebar.write('For more information about glaucoma, please visit the following websites:',
                                                                      'https://www.glaucoma.org/',
                                                                      'https://www.nei.nih.gov/eye-health/glaucoma'))
# Deploy the Streamlit app to the web
#if __name__ == '__main__':
 # st.deploy_app('deploy.py')