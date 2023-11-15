import argparse
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import keras
import tensorflow as tf

def prepare_image(file_path, target_size):
    """
    Prepare the image for prediction.

    Parameters:
    - file_path (str): Path to the image file.
    - target_size (tuple): The target size of the image (width, height).

    Returns:
    - img (numpy.ndarray): The processed image array.
    """
    img = load_img(file_path, target_size=target_size)  # Load the image
    img = img_to_array(img)  # Convert the image to a numpy array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = img / 255.0  # Scale pixel values to [0, 1]
    return img

def main():
    """
    Main script to classify an image using a pretrained CNN model.

    This script loads a saved Keras model and makes predictions on a user-specified image.
    The image is preprocessed to match the input format of the model before making the prediction.
    """
    # Setup argument parser to receive the image path from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to the image to be classified")
    args = parser.parse_args()
    
    # Load the saved model
    model = load_model('my_model.keras')

    # Prepare the image
    prepared_img = prepare_image(args.path, target_size=(28, 28))  # Ensure target_size is same as model's input size

    # Predict the class
    prediction = model.predict(prepared_img)
    predicted_class = np.argmax(prediction, axis=1)

    # Output the predicted class
    print(f"Predicted class: {predicted_class[0]}")

# Entry point of the script
if __name__ == "__main__":
    main()
