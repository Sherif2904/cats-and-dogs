import os
import tensorflow as tf
import numpy as np
import local_config as lc

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print('Loading Model..')
model = tf.keras.models.load_model(os.path.join(lc.MODEL_FILE))
model.trainable = False

def allowed_file(filename):
    '''
    Checks if a given file `filename` is of type image with 'png', 'jpg', or 'jpeg' extensions
    '''
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
    return (('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS))

def prepare_image(image_path):
    '''
    Loads image from the given `image_path` parameter
    '''
    image = load_img(image_path, target_size = (lc.SIZE, lc.SIZE))
    #--------------------
    # convert to array
    image = img_to_array(image)
    # reshape into a single sample with 3 channels
    image = image.reshape(1, 150, 150, 3)
    # center pixel data
    image = image.astype('float32')
    image = image / 255
    return image
    #-----------------------
    #return np.expand_dims(preprocess_input(img_to_array(image)), axis=0)

def make_prediction(filename):
    '''
    Predicts a given `filename` file
    '''
    print('Filename is ', filename)
    fullpath = os.path.join(lc.OUTPUT_DIR, filename)
    test_data = prepare_image(fullpath)
    predictions = model.predict(test_data)

    if predictions < 0.5:
        return str('Cat')
    else:
        return str('Dog')


