import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


model = keras.models.load_model('asl_model')

# visualization method
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
    plt.show()

# show_image('data/asl_images/b.png')

# loads and scales an image given the path
def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

image = load_and_scale_image('data/asl_images/b.png')
# plt.imshow(image, cmap='gray')
# plt.show()


# formatting and reshaping image
image = image_utils.img_to_array(image)
image = image.reshape(1,28,28,1) 

alphabet = "abcdefghiklmnopqrstuvwxy"

def predict_letter(file_path):
    # show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)
    # convert prediction to letter
    predicted_letter = alphabet[np.argmax(prediction)]
    return predicted_letter

print(predict_letter("data/asl_images/b.png"))
print(predict_letter("data/asl_images/a.png"))