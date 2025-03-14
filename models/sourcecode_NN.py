# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import drive
import zipfile
import requests
from tqdm.notebook import tqdm

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 224  # Standard size for many pre-trained models
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # Cloudy, Rainy, Sunny, Sunrise

# Path to zip file in Google Drive
zip_path = '/content/drive/MyDrive/weather_dataset.zip'

# Extract zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content')

# Path of dataset (adjust as needed)
DATA_DIR = '/content/Multi-class Weather Dataset'  # Change this if your data is elsewhere

def prepare_data(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Prepare and augment data for training and validation.
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found. Please check the path.")

    # Print classes available in the dataset
    classes = os.listdir(data_dir)
    print(f"Classes found: {classes}")

    # Print number of images per class
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            num_images = len(os.listdir(cls_path))
            print(f"Class {cls}: {num_images} images")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation data generator
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, valid_generator

def create_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    """
    Create a CNN model for weather image classification.
    Using transfer learning with MobileNetV2 as the base model.
    """
    # Load the pre-trained model without the top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    return model

def train_model(model, train_generator, valid_generator, epochs=EPOCHS):
    """
    Train the model with early stopping and learning rate reduction.
    """
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        'models/weather_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    return history, model

def evaluate_model(model, valid_generator):
    """
    Evaluate the model and generate performance metrics.
    """
    # Create directory for output if it doesn't exist
    os.makedirs('/content/output', exist_ok=True)

    # Predict classes
    valid_generator.reset()
    y_pred = model.predict(valid_generator, steps=int(np.ceil(valid_generator.samples / BATCH_SIZE)))
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Get true classes
    y_true = valid_generator.classes

    # Class names
    class_names = list(valid_generator.class_indices.keys())

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

    return cm, report, class_names

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('/content/output/training_history.png')
    plt.show()  # Display in Colab

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('/content/output/confusion_matrix.png')
    plt.show()  # Display in Colab

def fine_tune_model(model, train_generator, valid_generator, epochs=10):
    """
    Fine-tune the model by unfreezing some layers of the base model.
    """
    # Unfreeze the top layers of the base model
    for layer in model.layers[0].layers[-20:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with fine-tuning
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('/content/models/weather_cnn_model_finetuned.h5',
                           monitor='val_accuracy', save_best_only=True, mode='max')
        ]
    )

    return fine_tune_history, model

def save_model_to_drive(model_path, drive_path):
    """
    Save the model file to Google Drive.
    """
    # Create the destination directory if it doesn't exist
    drive_dir = os.path.dirname(drive_path)
    os.makedirs(drive_dir, exist_ok=True)

    # Copy the file
    tf.io.gfile.copy(model_path, drive_path, overwrite=True)
    print(f"Model saved to {drive_path}")

def test_single_image(model, image_path, img_size=IMG_SIZE):
    """
    Test the model on a single image.
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Get class names
    class_names = ['Cloudy', 'Rainy', 'Sunny', 'Sunrise']

    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    # Display the image and prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    # Display prediction probabilities for all classes
    plt.figure(figsize=(10, 3))
    plt.bar(class_names, prediction[0] * 100)
    plt.xlabel('Weather Class')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Probabilities')
    plt.ylim(0, 100)
    plt.show()

    return predicted_class, confidence

# Prepare data
print("Preparing data...")
train_generator, valid_generator = prepare_data()

# Create model
print("Creating model...")
model = create_model()

# Main execution

# Train model
print("Training model...")
history, model = train_model(model, train_generator, valid_generator)

# Plot training history
print("Plotting training history...")
plot_training_history(history)

# Fine-tune the model
print("Fine-tuning model...")
fine_tune_history, model = fine_tune_model(model, train_generator, valid_generator)

# Plot fine-tuning history
print("Plotting fine-tuning history...")
plot_training_history(fine_tune_history)

# Evaluate model
print("Evaluating model...")
cm, report, class_names = evaluate_model(model, valid_generator)

# Plot confusion matrix
print("Plotting confusion matrix...")
plot_confusion_matrix(cm, class_names)

# Save classification report
print("Saving classification report...")
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('/content/output/classification_report.csv')



print("Done!")


save_model_to_drive('/content/models/weather_cnn_model_finetuned.h5','/content/drive/MyDrive/models/weather_cnn_model_finetuned.h5')