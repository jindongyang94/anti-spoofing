"""
USAGE
python test.py video_demo --model=vgg16_pretrained.model --le=le.pickle --detector=face_detector
python test.py classify_images --location=/Users/jindongyang/Documents/repos/hubble/hubble_projects/hubble_spoofing_detection/data/external \
	--model=vgg16_pretrained.model --le=le.pickle --detector=face_detector
"""

import os
import pickle
import shutil
import time
from glob import glob
from os.path import abspath, dirname, join

import cv2
import fire
import imutils
import numpy as np
from imutils.video import VideoStream
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tqdm import tqdm

from modules.aws_helper import S3Helper
from modules.config import (DATALAKE_NAME, PROFILEIMG_FOLDER,
                            external_data_location, logger, models_location,
                            working_directory)


def video_demo(model, le, detector, confidence=0.5):
    """
    provide video live demo to check if the model works.
    """
    args = {
        'model': model,
        'detector': detector,
        'le': le,
        'confidence': confidence
    }

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.join("neural", "detectors",
                             args["detector"], "deploy.prototxt")
    modelPath = os.path.join(
        "neural", "detectors", args["detector"], "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    classifiermodelpath = os.path.join('neural', 'models', args['model'])
    model = load_model(classifiermodelpath)
    le = pickle.loads(open(os.path.join('neural', args["le"]), "rb").read())

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        frame = vs.read()
        frame, _, _ = label(frame, net, model, le, args['confidence'])
        # show the output frame and wait for a key press
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def classify_images(location, detector, model, le, confidence=0.9):
    """
    From a image folder location:
    1. Create a real and fake image folder in the current image folder itself. (Only if there aren't such a folder)
    2. Classify the images into real and fake and store them within the created folders. 
    """

    args = {
        'detector': detector,
        'model': model,
        'le': le
    }

    # Create Folders
    real_location = os.path.join(external_data_location, location, 'real')
    fake_location = os.path.join(external_data_location, location, 'fake')
    noface_location = os.path.join(external_data_location, location, 'noface')
    if not glob(real_location):
        os.mkdir(real_location)
    if not glob(fake_location):
        os.mkdir(fake_location)
    if not glob(noface_location):
        os.mkdir(noface_location)

    # Load Models
    # Load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.join(models_location, "detectors",
                             args["detector"], "deploy.prototxt")
    modelPath = os.path.join(models_location, "detectors",
                             args["detector"], "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    classifiermodelpath = os.path.join(
        models_location, 'models', args['model'])
    model = load_model(classifiermodelpath)
    le = pickle.loads(
        open(os.path.join(models_location, 'labels', args["le"]), "rb").read())

    # Grab all images from given folder
    unsorted_folder = os.path.join(
        external_data_location, location, 'unsorted')
    images = glob(os.path.join(unsorted_folder, '*.png'))
    jpg_images = glob(os.path.join(unsorted_folder, '*.jpg'))
    images.extend(jpg_images)

    # Maintain counters for all types of images
    real_counter = 0
    fake_counter = 0
    noface_counter = 0

    bar = tqdm(images, dynamic_ncols=True, desc='Bar desc', leave=True)
    for image in bar:
        frame = cv2.imread(image)
        frame, contains_fake, detected_faces = label(
            frame, net, model, le, confidence)

        # Relocate the image based on whether it is fake, real or noface
        image_name = os.path.basename(image)
        if detected_faces == 0:
            image_location = os.path.join(noface_location, image_name)
            noface_counter += 1
        elif contains_fake:
            image_location = os.path.join(fake_location, image_name)
            fake_counter += 1
        else:
            image_location = os.path.join(real_location, image_name)
            real_counter += 1

        # Shift image to classified location
        cv2.imwrite(image_location, frame)

        # Delete image from unsorted location
        # os.remove(image)

        image_folder_location = os.path.split(image_location)[0]
        image_category = os.path.split(image_folder_location)[1]
        bar.set_description(os.path.join(image_category, image_name))
        bar.refresh()

    logger.info('Real Images Classified: %s' % real_counter)
    logger.info('Fake Images Classified: %s' % fake_counter)
    logger.info('No Face Images Classified: %s' % noface_counter)

    # Count present images in each folder location
    total_real = len(glob(os.path.join(real_location, '*')))
    total_fake = len(glob(os.path.join(fake_location, '*')))
    total_noface = len(glob(os.path.join(noface_location, '*')))

    logger.info('Real Images Present: %s' % total_real)
    logger.info('Fake Images Present: %s' % total_fake)
    logger.info('No Face Images Present: %s' % total_noface)


def classify_images_s3_local(s3bucket, s3folderpath, detector, model, le, confidence=0.5):
    """
    This is simplified function where we will:
    1. Download the s3 photos
    2. Classify the photos locally
    3. Delete the photos that are not spoofed

    Repeat the process till all s3 images are done.
    """
    pass


def classify_images_s3(s3bucket, s3folderpath, detector, model, le, confidence=0.5):
    """
    This function will take in the location to the current s3 image folder, and separate the images within into folders within the given folder
    i.e. s3 image folder --> hubble-datalake/images/profilepics
             real images --> hubble-datalake/images/profilepics/real
             fake images --> hubble-datalake/images/profilepics/fake

    """
    pass


# --------------------------------------------------------------------------------------------------------------------------
def label(frame, net, model, le, confidence):
    """
    Classify an image based on model given.
    """
    frame = imutils.resize(frame, width=600)

    # flip the frame
    frame = cv2.flip(frame, 1)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # maintain a label if the image is false or not
    contains_fake = False

    # maintain number of faces detected
    detected_faces = 0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        detected_confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if detected_confidence > confidence:
            detected_faces += 1
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            try:
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
            except:
                continue

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            # draw the label and bounding box on the frame
            if label == 'real':
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
            else:
                contains_fake = True
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

    return frame, contains_fake, detected_faces


def find_datafolder():
    """
    Return parent folder, unsorted folder, real folder and spoof folder
    """
    original_folder = join(abspath('.'), 'data')

    unsorted_folder = join(original_folder, 'unsorted')
    category1_folder = join(original_folder, 'real')
    category2_folder = join(original_folder, 'spoof')

    return original_folder, unsorted_folder, category1_folder, category2_folder


if __name__ == "__main__":
    fire.Fire()
