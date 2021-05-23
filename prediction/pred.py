import tensorflow as tf
from keras_preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

model = tf.keras.models.load_model('../model/my_model.h5', custom_objects=None, compile=True, options=None)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(.3, 1.),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

img_size = (224, 224)
random_seed = 123

training_set = train_datagen.flow_from_directory(
    '../dataset',
    color_mode='rgb',
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    seed=random_seed,
    subset='training'
)

testing_set = train_datagen.flow_from_directory(
    '../dataset',
    target_size=img_size,
    batch_size=16,
    class_mode='binary',
    seed=random_seed,
    subset='validation'
)

loss, acc = model.evaluate(training_set)


def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)

    print(decode_predictions(prediction, top=3)[0])


classify("../dataset/pred/pred20.jpg")
