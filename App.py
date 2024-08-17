import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the ResNet50 model without the top layer (FC layer) and with pre-trained ImageNet weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add Global Max Pooling layer after the model to flatten the output
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Check if GPU is available
if tensorflow.config.list_physical_devices('GPU'):
    logging.info("Using GPU for computation")
else:
    logging.info("Using CPU for computation")

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Define the directory path
image_directory = 'images'

# Check if the directory exists, if not create it
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Filter valid image files (jpg, jpeg, png) from the directory
valid_extensions = ['.jpg', '.jpeg', '.png']
filenames = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if os.path.splitext(file)[1].lower() in valid_extensions]

# Log the number of files found
logging.info(f"Found {len(filenames)} valid image files.")

# List to store extracted features
feature_list = []

# Loop over all image files and extract features
for file in tqdm(filenames):
    try:
        features = extract_features(file, model)
        feature_list.append(features)
        logging.info(f"Successfully processed {file}")
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

# Generate a timestamp for unique file names
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Save the extracted features and filenames to pickle files
pickle.dump(feature_list, open(f'embeddings_{timestamp}.pkl', 'wb'))
pickle.dump(filenames, open(f'filenames_{timestamp}.pkl', 'wb'))

# Save the model using the native Keras format
model.save(f'resnet_model_{timestamp}.keras')

logging.info("Feature extraction completed and data saved.")
