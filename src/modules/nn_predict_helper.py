import os

import cv2
from math import ceil
import imutils
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import img_to_array


def load_model(weights_path ,img_width, img_height, depth, classes):
    """
    We will only use this function if there is a need to load only weights instead of .model file. 
    """
    inputShape = (img_height, img_width, depth)
    #VGG-16 model
    model = Sequential()
   
    model.add(ZeroPadding2D((1, 1), input_shape=inputShape))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Top-model for anti-spoofing
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    
    model.add(top_model)
    
    if weights_path:
        model.load_weights(weights_path)
    else:
        print ('Could not load model!')
    
    return model

# --------------------------------------------------------------------------------------------------------------------------
# Label Functions
# --------------------------------------------------------------------------------------------------------------------------
def label_with_face_detector_original(frame, net, model, le, confidence):
    """
    Classify an image based on model given.
    """
    detection_threshold = 0.7

    # modify the frame if needed
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
        if detected_confidence > detection_threshold:
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

def label_with_face_detector_ultra(frame, net, model, le, confidence):
    """
    Classify an image based on model given.
    """
    # Parameters
    input_size = (320, 340)
    image_std = 128
    iou_threshold = 0.3
    center_variance = 0.1
    size_variance = 0.2
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0],
                [64.0, 96.0], [128.0, 192.0, 256.0]]
    strides = [8.0, 16.0, 32.0, 64.0]
    priors = define_img_size(input_size)
    detection_threshold = 0.7

    # Transform the frame if needed for video
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    rect = cv2.resize(frame, input_size)
    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rect, 1 / image_std, input_size, 127)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    boxes, scores = net.forward(["boxes", "scores"])

    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(frame.shape[1], frame.shape[0], scores, boxes, detection_threshold)

    # maintain a label if the image is false or not
    contains_fake = False

    # maintain number of faces detected
    detected_faces = 0

    # loop over the detections
    for i in range(0, boxes.shape[0]):

        detected_faces += 1
        # compute the (x, y)-coordinates of the bounding box for
        # the face and extract the face ROI
        box = boxes[i, :]
        # cv2.rectangle(frame, (box[0], box[1]),
        #                 (box[2], box[3]), (0, 255, 0), 2)
        (startX, startY, endX, endY) = box.astype("int")

        # ensure the detected bounding box does fall outside the
        # dimensions of the frame
        startX = max(0, startX-10)
        startY = max(0, startY-10)
        endX = min(w, endX+10)
        endY = min(h, endY+10)

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

# --------------------------------------------------------------------------------------------------------------------------
# Sub Functions
# --------------------------------------------------------------------------------------------------------------------------
def define_img_size(image_size):
    # Parameters
    strides = [8.0, 16.0, 32.0, 64.0]
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0],
             [64.0, 96.0], [128.0, 192.0, 256.0]]

    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list,
                             shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    # print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance *
        priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)