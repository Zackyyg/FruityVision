import argparse
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

model = load_model('bruh/fresh_classifier.h5')

class_indices = {
    0: 'freshapples',
    1: 'freshbanana',
    2: 'freshoranges',
    3: 'rottenapples',
    4: 'rottenbanana',
    5: 'rottenoranges'
}

parser = argparse.ArgumentParser(description='Classify an image as fresh or rotten fruit.')
parser.add_argument('image_path', type=str, help='Path to the image file to classify.')
args = parser.parse_args()

def prepare_image(file_path):
    img = load_img(file_path, target_size=(150, 150))  # Adjust the size to match your model's input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255  # Normalize the pixel values as you did during training
    return img_array

def predict_image(file_path):
    prepared_image = prepare_image(file_path)
    prediction = model.predict(prepared_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_indices[predicted_class_index]
    return predicted_class

if os.path.isfile(args.image_path):
    result = predict_image(args.image_path)
    print(result)
else:
    print("The file specified does not exist.")
