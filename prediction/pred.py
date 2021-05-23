import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob


print(tf.__version__)

model = tf.keras.models.load_model(
    '../model/my_model.h5',
    custom_objects=None,
    compile=True,
    options=None)

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

img_size = (150, 150)

training_set = train_datagen.flow_from_directory(
    '../dataset',
    target_size=img_size,
    class_mode='binary',
    subset='training'
)

testing_set = train_datagen.flow_from_directory(
    '../dataset',
    target_size=img_size,
    class_mode='binary',
    subset='validation'
)

loss, acc = model.evaluate(training_set)

img_dir = "/path/to/folder/where/prediction/images/are"
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
results = []

for file in files:
    test_images = image.load_img(file, target_size=img_size)
    test_images = image.img_to_array(test_images)
    test_images = np.expand_dims(test_images, axis=0)
    images = np.vstack([test_images])
    classes = model.predict(images, batch_size=10)
    classes = np.round(classes)
    data.append(file)
    results.append(classes)

for x, y in zip(data, results):
    print(x, y)

L = 4, W = 4
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, L * W):
    img = cv2.imread(data[i])
    axes[i].imshow(img)
    axes[i].set_title(results[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])

fig.tight_layout()
fig.savefig('/path/to/save/png', bbox_inches='tight')
