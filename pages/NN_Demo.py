import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="Neural Network Demo", page_icon="🧠", layout="wide")

# Title and description
st.title("Weather Image Classification with Neural Network")
st.write("""
This demo allows you to upload a weather image and the neural network model will
classify it into one of the following categories: Cloudy, Rainy, Sunny, or Sunrise.
""")

# Load the trained model
@st.cache_resource
def load_nn_model():
    try:
        # Try to load the fine-tuned model first
        model_path = os.path.join('models', 'weather_cnn_model_finetuned.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            # Fall back to the regular model
            model_path = os.path.join('models', 'weather_cnn_model.h5')
            model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image preprocessing function
def preprocess_image(img):
    # Resize the image
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to make prediction
def predict_weather(img_array, model):
    # Class names
    class_names = ['Cloudy', 'Rainy', 'Sunny', 'Sunrise']
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100
    
    return predicted_class, confidence, prediction[0]

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose a weather image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess the image
            img_array = preprocess_image(img)
            
            # Make prediction when user clicks the button
            if st.button('Predict Weather'):
                with st.spinner('Analyzing image...'):
                    # Load model
                    model = load_nn_model()
                    
                    if model:
                        predicted_class, confidence, prediction_scores = predict_weather(img_array, model)
                        
                        # Display prediction
                        st.success(f"Prediction: **{predicted_class}**")
                        st.info(f"Confidence: {confidence:.2f}%")
                        
                        # Create and display a bar chart of prediction probabilities
                        class_names = ['Cloudy', 'Rainy', 'Sunny', 'Sunrise']
                        fig, ax = plt.subplots()
                        y_pos = np.arange(len(class_names))
                        ax.barh(y_pos, prediction_scores, align='center')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(class_names)
                        ax.invert_yaxis()  # Labels read top-to-bottom
                        ax.set_xlabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        
                        # Display the chart
                        st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing image: {e}")

with col2:
    st.subheader("Model Performance")
    
    # Show model architecture information
    st.write("### Model Architecture")
    st.write("""
    The neural network model used for this weather image classification task is based on MobileNetV2, 
    a pre-trained convolutional neural network optimized for mobile devices. The model was fine-tuned 
    on our weather dataset to achieve high accuracy with efficient computation.
    
    **Key components:**
    - Pre-trained MobileNetV2 base (transfer learning)
    - Global Average Pooling layer
    - Dense layers with dropout for regularization
    - Softmax activation for multi-class classification
    """)
    
    # Try to load and display performance metrics
    try:
        if os.path.exists('data/training_history.png'):
            st.write("### Training History")
            st.image('data/training_history.png', use_column_width=True)
        
        if os.path.exists('data/confusion_matrix.png'):
            st.write("### Confusion Matrix")
            st.image('data/confusion_matrix.png', use_column_width=True)
    except Exception as e:
        st.warning("Model performance visualizations not available yet. Train the model first.")
    
    # Model usage instructions
    st.write("### How to Use")
    st.write("""
    1. Upload a weather image using the file uploader on the left
    2. Click the "Predict Weather" button
    3. View the predicted weather class and confidence score
    4. The bar chart shows probabilities for all weather categories
    """)

# Display sample images
st.subheader("Sample Images")
st.write("If you don't have a weather image handy, here are some sample weather conditions:")

# Create a layout for sample images
sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

# Placeholder for sample images (replace these with actual paths when deployed)
with sample_col1:
    st.write("Cloudy")
    st.caption("Example of a cloudy sky")

with sample_col2:
    st.write("Rainy")
    st.caption("Example of rainy weather")

with sample_col3:
    st.write("Sunny")
    st.caption("Example of a sunny day")

with sample_col4:
    st.write("Sunrise")
    st.caption("Example of a sunrise")

# Technical details in an expandable section
with st.expander("Technical Details"):
    st.write("""
    ### Model Details
    - **Base Model**: MobileNetV2 (pre-trained on ImageNet)
    - **Input Shape**: 224x224x3 (RGB images)
    - **Training Strategy**: Transfer learning with fine-tuning
    - **Optimization**: Adam optimizer with learning rate reduction
    - **Regularization**: Dropout layers to prevent overfitting
    - **Data Augmentation**: Rotation, zoom, shift, flip for training data
    
    ### Preprocessing
    Images are preprocessed in the following steps:
    1. Resize to 224x224 pixels
    2. Normalize pixel values to range [0-1]
    3. Apply augmentation during training (not during inference)
    
    ### Performance Considerations
    - The model balances accuracy and inference speed
    - Fine-tuning improves performance on specific weather conditions
    - Confidence scores help determine prediction reliability
    """)

# Add footer
st.markdown("---")
st.caption("Weather Image Classification Neural Network Demo • IS 2567-2 Final Project")