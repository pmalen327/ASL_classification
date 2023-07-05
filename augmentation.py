import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import IPython


# load data
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

# split values
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# split images
x_train = train_df.values
x_valid = valid_df.values

# make binary categories
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# normalize
x_train = x_train / 255
x_valid = x_valid / 255

# reshape
x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)


# creating model
model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))

# randomizes images to supplement data set
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True, 
    vertical_flip=False,
)  

# batching
batch_size = 32
img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)

# fit and compile
datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# augmentation fit and train
model.fit(img_iter,
          epochs=20,
          verbose=2,
          steps_per_epoch=len(x_train)/batch_size,
          validation_data=(x_valid, y_valid))

model.save('asl_model')
model.summary()