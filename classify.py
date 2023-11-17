import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img

model = load_model('./fresh_classifier.h5')

class_indices = {
    0: 'freshapples',
    1: 'freshbanana',
    2: 'freshoranges',
    3: 'rottenapples',
    4: 'rottenbanana',
    5: 'rottenoranges'
}

def prepare_image(file_path):
    img = load_img(file_path, target_size=(150, 150))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255

def predict_image(file_path):
    prepared_image = prepare_image(file_path)
    prediction = model.predict(prepared_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_indices[predicted_class_index]
    return predicted_class

image_path = 'download (1).jpeg'
result = predict_image(image_path)
print(result)
