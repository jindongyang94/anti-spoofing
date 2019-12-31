"""
USAGE:
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
from modules.config import (DATALAKE_NAME, DETECTORS_DIR, EXTERNAL_DATA_DIR,
                            INTERIM_DATA_DIR, LABELS_DIR, NN_MODELS_DIR,
                            PROFILEIMG_FOLDER, WORKING_DIR, find_model, logger)
from modules.nn_predict_helper import (label_with_face_detector_original,
                                       label_with_face_detector_ultra)


def video_demo(model, le, detector, confidence=0.9):
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
    face_detector_path = os.path.join(DETECTORS_DIR, args['detector'])
    protoPath = find_model(face_detector_path, 'prototxt')
    modelPath = find_model(face_detector_path, "caffemodel")     
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    classifiermodelpath = os.path.join(
        NN_MODELS_DIR, args['model'])
    model = load_model(classifiermodelpath)
    le = pickle.loads(
        open(os.path.join(LABELS_DIR, args["le"]), "rb").read())

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        frame = vs.read()
        if args['detector'] == 'face_RFB':
            frame, _, _ = label_with_face_detector_ultra(frame, net, model, le, args['confidence'])
        else:
            frame, _, _ = label_with_face_detector_original(frame, net, model, le, args['confidence'])
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
    real_location = os.path.join(INTERIM_DATA_DIR, location, 'real')
    fake_location = os.path.join(INTERIM_DATA_DIR, location, 'fake')
    noface_location = os.path.join(INTERIM_DATA_DIR, location, 'noface')
    if not glob(real_location):
        os.mkdir(real_location)
    if not glob(fake_location):
        os.mkdir(fake_location)
    if not glob(noface_location):
        os.mkdir(noface_location)

    # Load Models
    # Load our serialized face detector from disk
    print("[INFO] loading face detector...")
    face_detector_path = os.path.join(DETECTORS_DIR, args['detector'])
    protoPath = find_model(face_detector_path, 'prototxt')
    modelPath = find_model(face_detector_path, "caffemodel")     
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    classifiermodelpath = os.path.join(NN_MODELS_DIR, args['model'])
    model = load_model(classifiermodelpath)
    le = pickle.loads(
        open(os.path.join(LABELS_DIR, args["le"]), "rb").read())

    # Grab all images from given folder
    unsorted_folder = os.path.join(EXTERNAL_DATA_DIR, location)
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
        if args['detector'] == 'face_RFB':
            frame, finally_fake, detected_faces = label_with_face_detector_ultra(frame, net, model, le, args['confidence'])
        else:
            frame, finally_fake, detected_faces = label_with_face_detector_original(frame, net, model, le, args['confidence'])

        # Relocate the image based on whether it is fake, real or noface
        image_name = os.path.basename(image)
        if detected_faces == 0:
            image_location = os.path.join(noface_location, image_name)
            noface_counter += 1
        elif finally_fake:
            image_location = os.path.join(fake_location, image_name)
            fake_counter += 1
        else:
            image_location = os.path.join(real_location, image_name)
            real_counter += 1

        # Shift image to classified location
        cv2.imwrite(image_location, frame)

        # Delete image from unsorted location
        os.remove(image)

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
# Sub Functions
# --------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    fire.Fire()
