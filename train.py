from pathlib import Path
from shutil import copy
import os
import random

import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input


def split_data(train=0.7, validation=0.2, test=0.1):
    '''Splits the data stored in data/labeled and stores it in data/train
    according to the ratios for training, validation, and test data.'''
    assert (train + validation + test) - 1 < 1e-8
    # Find all images in data/labeled and its subdirectories
    total_size = 0
    for dirpath, dirnames, filenames in os.walk("data/labeled/"):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            # Randomly decide whether an instance is used to train, validate, or test
            rand = random.uniform(0, 1)
            label = dirpath.split('/')[-1]
            if rand < train:
                data_type = 'train'
            elif rand < train+validation:
                data_type = 'validation'
            else:
                data_type = 'test'
            path = 'data/CNN/%s/%s/' % (data_type, label)
            # Make a directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            copy(dirpath+'/'+filename, path)


def make_datagen():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)

    # the validation data should not be augmented!
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return train_datagen, test_datagen


def make_model():
    base_model = InceptionResNetV2(
        include_top=False, weights='imagenet', input_shape=(150, 300, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(7, activation='softmax')(x)  # New softmax layer
    model = models.Model(inputs=base_model.input, outputs=predictions)

    return model


def make_generators(train_datagen, test_datagen, batch_size=16):
    # this is a generator that will read pictures found in
    # subfolers of 'data/CNN/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        'data/CNN/train',  # this is the target directory
        target_size=(150, 300),  # all images will be resized to 150x300
        batch_size=batch_size,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        'data/CNN/validation',
        target_size=(150, 300),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator


if __name__ == '__main__':
    split_data(0.7, 0.2, 0.1)
    print("Done splitting data!")

    train_datagen, test_datagen = make_datagen()
    model = make_model()
    # we chose to train the top 2 inception blocks
    # we will freeze the first 249 layers and unfreeze the rest
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    batch_size = 16
    adam = Adam(lr=0.0001)
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    train_generator, validation_generator = make_generators(
        train_datagen, test_datagen, batch_size=batch_size)
    mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',
                                         save_weights_only=True, save_freq=3)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[mc])
    # always save your weights after training or during training
    model.save_weights('first_try.h5')
