"""
USAGE:
python train.py train --dataset=portrait_videos --model=vgg16_pretrained.model --le=le.pickle
"""

import argparse
import os
import pickle

import cv2
import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras import callbacks
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.config import (FIGURES_DIR, LABELS_DIR, NN_MODELS_DIR,
                            PROCESSED_DATA_DIR)
from modules.nn_train_helper import Model

# set the matplotlib backend so figures can be saved in the background
matplotlib.use('TkAgg')


def train(dataset, model, le, plot='plot.png'):
    """
    Function Wrapper to initialize training using Keras Model.

    Training can train by taking all pictures from all subfolders, but please make sure that the labels are CONSISTENT.
    Make sure that the direct folder name is also the label name as well. 
    """
    # if dataset == all, we take all folders in the PROCESSED_DATA_DIR
    if dataset == 'all':
        dataset = PROCESSED_DATA_DIR

    args = {
        'dataset': dataset,
        'model': model,
        'le': le,
        'plot': plot
    }

    # initialize the initial learning rate, batch size, and number of
    # epochs to train for
    INIT_LR = 1e-4
    BS = 8
    EPOCHS = 100

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    dataset_path = os.path.join(PROCESSED_DATA_DIR, args['dataset'])
    image_paths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    for image_path in image_paths:
        # extract the class label from the filename, load the image and
        # resize it to be a fixed 32x32 pixels, ignoring aspect ratio
        label = image_path.split(os.path.sep)[-2]
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))
        except Exception as e:
            print("Image %s is broken. Continue.." %
                  image_path.split(os.path.sep)[-2:])
            continue

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data into a NumPy array, then preprocess it by scaling
    # all pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    # encode the labels (which are currently strings) as integers and then
    # one-hot encode them
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np_utils.to_categorical(labels, 2)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=20,
                             zoom_range=0.15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                             # vertical_flip=True,
                             fill_mode="nearest"
                             )

    # initialize the optimizer and model
    # print("[INFO] compiling model...")
    # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # model_builder = Model(width=32, height=32, depth=3, classes=len(le.classes_))
    # model = model_builder.build_liveness()

    # VGG optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model_builder = Model(width=32, height=32, depth=3,
                          classes=len(le.classes_))
    model = model_builder.build_VGG()
    nFreeze = 10
    # freeze layers
    for layer in model.layers[:nFreeze]:
        layer.trainable = False

    # model compilation
    model.compile(loss="binary_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network for {} epochs...".format(EPOCHS))
    earlyStopping = callbacks.EarlyStopping(monitor='val_acc',
                                            patience=10,
                                            verbose=0, mode='auto')
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        callbacks=[earlyStopping],
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS
    )

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=BS)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=le.classes_))

    # save the network to disk
    print("[INFO] serializing network to '{}'...".format(args["model"]))
    model_path = os.path.join(NN_MODELS_DIR, args['model'])
    model.save(model_path)

    # save the label encoder to disk
    le_path = os.path.join(LABELS_DIR, args['le'])
    f = open(le_path, "wb")
    f.write(pickle.dumps(le))
    f.close()

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    # print(H.history.keys())
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plot_path = os.path.join(FIGURES_DIR, args["plot"])
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    fire.Fire()
