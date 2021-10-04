# This module trains the CNN based on the labels provided in ./data/CNN
# Note that data must be first split into train, validation, and test data
# by running split_data.py.
# Reference:
# https://towardsdatascience.com/a-single-function-to-streamline-image-classification-with-keras-bd04f5cfe6df

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
import cv2
import json
import os


NUM_EPOCHS = 10
BATCH_SIZE = 16
DATA_FOLDER = './data/CNN/'


def create_generators(folderpath=DATA_FOLDER):
    '''Creates flow generators to supply images one by one during 
    training/validation phases. Useful when working with large datasets 
    that can't be directly loaded into the memory.'''
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        folderpath+'train',  # This is the source directory for training images
        target_size=(300, 150),  # All images will be resized to 300 x 150
        batch_size=BATCH_SIZE,
        # Specify the classes explicitly
        classes=['Bishop_Black', 'Bishop_White', 'Empty', 'King_Black', 'King_White', 'Knight_Black',
                 'Knight_White', 'Pawn_Black', 'Pawn_White', 'Queen_Black', 'Queen_White', 'Rook_Black', 'Rook_White'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
    # Follow the same steps for validation generator
    validation_datagen = ImageDataGenerator(rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(
        folderpath+'validation',
        target_size=(300, 150),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    return (train_generator, validation_generator)


def create_model(optimizer=RMSprop(learning_rate=0.001)):
    '''Creates a CNN architecture and compiles it.'''
    model = Sequential([
        # Note the input shape is the desired size of the image 300 x 150 with 3 bytes color
        # The first convolution
        Conv2D(16, (3, 3), activation='relu', input_shape=(300, 150, 3)),
        MaxPooling2D(2, 2),
        # The second convolution
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # The third convolution
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # The fourth convolution
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # The fifth convolution
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        # Flatten the results to feed into a dense layer
        Flatten(),
        # 128 neuron in the fully-connected layer
        Dense(128, activation='relu'),
        # 13 output neurons for 13 classes with the softmax activation
        Dense(13, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


def fit_model(model, train_generator, validation_generator, callbacks=[], save=False, filename=""):
    '''Given the model and generators, trains the model and saves weights if
    needed. Callbacks can be provided to save intermediate results.
    Returns a history of model's performance (for plotting purpose).'''

    total_sample = train_generator.n

    history = model.fit(
        train_generator,
        steps_per_epoch=int(total_sample/BATCH_SIZE),
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        callbacks=callbacks)

    if save:
        model.save_weights(filename)

    return history


def plot_accuracy(history):
    '''Given training history, plots accuracy of a model.'''
    plt.figure(figsize=(7, 4))
    plt.plot([i+1 for i in range(NUM_EPOCHS)],
             history.history['acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training accuracy with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def plot_loss(history):
    '''Given training history, plots loss of a model.'''
    plt.figure(figsize=(7, 4))
    plt.plot([i+1 for i in range(NUM_EPOCHS)],
             history.history['loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def save_history(history, filename="./history.json"):
    '''Saves the given training history as a .json file.'''
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(filename, 'w'))


def load_history(filename="./history.json"):
    '''Loads training history from the path to a .json file. Returns a dict.'''
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def test_model(model):
    '''Tests the given model on the test set and prints its accuracy.
    Does not return anything.'''
    testdir = DATA_FOLDER + 'test'

    # pieces = ['Empty', 'Rook', 'Knight', 'Bishop', 'Queen', 'Pawn', 'King']
    pieces = ['Empty', 'Rook_White', 'Rook_Black', 'Knight_White', 'Knight_Black', 'Bishop_White',
              'Bishop_Black', 'Queen_White', 'Queen_Black', 'King_White', 'King_Black', 'Pawn_White', 'Pawn_Black']
    pieces.sort()
    score = 0
    total_size = 0
    for subdir, dirs, files in os.walk(testdir):
        for file in files:
            if file == ".DS_Store":
                continue
            piece = subdir.split('/')[-1]
            path = os.path.join(subdir, file)
            y_prob = model.predict(cv2.imread(path).reshape(1, 300, 150, 3))
            y_pred = y_prob.argmax()
            if y_pred < 0 or y_pred >= len(pieces):
                print(y_pred, y_prob)
            if piece == pieces[y_pred]:
                score += 1
            total_size += 1
    print("TEST SET ACCURACY:", score/total_size)


if __name__ == '__main__':
    train_generator, validation_generator = create_generators(DATA_FOLDER)
    model = create_model()
    history = fit_model(model, train_generator,
                        validation_generator, save=False)
    save_history(history, "./history.json")
    plot_accuracy(history)
    plot_loss(history)
    test_model(model)
    model.save_weights('./model_weights.h5')
