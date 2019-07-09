#!/usr/bin/env python
# coding: utf-8

# # General Purpose Defect Inspection
# ## Experiment with a custom covolutional neural network (CNN) and pre-trained CNN using a dataset of M&Ms to detect various defects.

import keras
from keras.models import Sequential,load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

# directory of M&M images
img_dir = '../../data/crop_mm'
# the label csv file
df=pd.read_csv('../../data/crop_mm_relabel.csv')
# columns are the labels, or type of defects.
columns=df.columns.tolist()[1:]

image_size = 160
batch_size = 16

train_datagen=ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True,rotation_range=180)
test_datagen=ImageDataGenerator(rescale=1./255.)

train_generator=train_datagen.flow_from_dataframe(
dataframe=df[:round(df.shape[0]*0.9)],
directory=img_dir,
x_col="External ID",
y_col=columns,
batch_size=batch_size,
seed=1,
shuffle=True,
class_mode="other",
target_size=(image_size,image_size))

valid_generator=test_datagen.flow_from_dataframe(
dataframe=df[round(df.shape[0]*0.9):],
directory=img_dir,
x_col="External ID",
y_col=columns,
batch_size=batch_size,
shuffle=False,
class_mode="other",
target_size=(image_size,image_size))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# Build a custom CNN with a few Conv blocks

callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='val_acc',
patience=50,
),
keras.callbacks.ModelCheckpoint(
filepath='mm_scratch.h5',
monitor='val_acc',
save_best_only=True,
),
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss',
factor=0.5,
patience=50,
)
]

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(columns), activation='sigmoid'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=500,
                    callbacks=callbacks_list,
)

model = load_model('mm_scratch.h5')

valid_generator.reset()
test_loss, test_acc = model.evaluate_generator(valid_generator, steps = STEP_SIZE_VALID)
print('best acc of CNN tained from scratch:', test_acc)

# Create the base model from the pre-trained model

callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='val_acc',
patience=50,
),
keras.callbacks.ModelCheckpoint(
filepath='mm_transfer.h5',
monitor='val_acc',
save_best_only=True,
),
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss',
factor=0.5,
patience=50,
)
]

base_model = keras.applications.InceptionV3(input_shape=(image_size, image_size, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(len(columns), activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=500,
                    callbacks=callbacks_list,
)

model = load_model('mm_transfer.h5')

base_model.trainable = True
# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False


model.compile(loss='binary_crossentropy',
              optimizer = keras.optimizers.RMSprop(lr=0.0001),
              metrics=['accuracy'])

history_fine = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=500,
                    callbacks=callbacks_list,
)


model = load_model('mm_transfer.h5')

valid_generator.reset()
test_loss, test_acc = model.evaluate_generator(valid_generator, steps = STEP_SIZE_VALID)
print('best acc of model trained with transfer learning:', test_acc)


# ### References:
# Deep Learning with Python by François Chollet
#
# Tutorial on Keras ImageDataGenerator with flow_from_dataframe by Vijayabhaskar J
