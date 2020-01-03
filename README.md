# hubble_spoofing_detection

==============================

detecting spoofs from real attendance photos

## Project Organization

------------

    .
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data
    │   ├── external
    │   │   ├── attendance_photos
    │   │   └── portrait_videos
    │   │       ├── fake
    │   │       └── real
    │   ├── interim
    │   │   ├── attendance_photos
    │   │   │   ├── fake
    │   │   │   ├── noface
    │   │   │   └── real
    │   │   └── portrait_videos
    │   │       ├── fake
    │   │       └── real
    │   └── processed
    │       ├── attendance_photos
    │       └── portrait_videos
    │           ├── fake
    │           └── real
    ├── docs
    │   ├── Makefile
    │   ├── _build
    │   │   ├── doctrees
    │   │   │   └── environment.pickle
    │   │   └── html
    │   │       ├── _sources
    │   │       ├── _static
    │   │       └── objects.inv
    │   ├── conf.py
    │   └── make.bat
    ├── models
    │   ├── detectors
    │   │   ├── face_detector
    │   │   │   ├── deploy.prototxt
    │   │   │   └── res10_300x300_ssd_iter_140000.caffemodel
    │   │   └── haarcascade_eye.xml
    │   ├── labels
    │   │   └── le.pickle
    │   ├── nn_models
    │   │   └── vgg16_pretrained.model
    │   └── nn_pretrained_weights
    │       ├── 3DMAD-ftweights18.h5
    │       ├── REPLAY-ftweights18.h5
    │       └── vgg16_weights.h5
    ├── notebooks
    ├── references
    ├── reports
    │   └── figures
    ├── requirements.txt
    ├── setup.py
    ├── src
    │   ├── __init__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── make_dataset_aws.py
    │   │   └── make_dataset_video.py
    │   ├── features
    │   │   ├── __init__.py
    │   │   └── label_dataset.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── predict.py
    │   │   └── train.py
    │   ├── modules
    │   │   ├── __init__.py
    │   │   ├── aws_helper.py
    │   │   ├── config.py
    │   │   ├── nn_predict_helper.py
    │   │   └── nn_train_helper.py
    │   └── visualization
    │       ├── __init__.py
    │       ├── filters.py
    │       └── visualize.py
    ├── test_environment.py
    └── tox.ini

------------

Credits goes to <https://www.pyimagesearch.com/author/adrian/>

This library is simply to show how you can train your own simple face detection wtih anti-spoofing function:  

## Processing Dataset

### Process a video into frames

1. Make a video of yourself for about 20 -30 seconds and save it under the <data/external/portrait_videos/real> folder.  
2. Take a video of the video of yourself and save it under the <data/external/portrait_videos/fake> folder.  
3. Split the videos to frames or photos to create your dataset using function:  `make data_video location=portrait_videos detector=face_detector skip=8 reset=0`. The frames will be saved under <data/processed/portrait_videos> folder. The reset variable is useful in determing if you want to delete all existing frames from the folder or not.  

### Process images based on static images

1. Add the real photos in a new folder under <data/processed/new_folder/real>
2. Add the fake photos in a new folder under <data/processed/new_folder/fake>

Examples of the function commands are pasted at the top of all scripts.  
Use the makeFile functions instead of interacting with the scripts yourself.

### Preprocess images based on unclassified images

1. Add all the unclassified photos in a new folder under <data/external/new_folder>  
2. Run `make data_label location=new_folder`. What this will do is that it will create the necessary folders needed in the <data/processed> folder, and move the images to the designated label.  
3. Alternatively, you can classify the images using a pretrained model: `make classify location=attendance_photos`. You can change the model, face_detector and labels in the Makefile.  

## Training

There is not much change needed here except the understanding of some parameters given.

1. Train the model with ALL the available data using `make train location=all model=vgg16_pretrained.model label=le.pickle`.  
You can use only one location if you specify the folder at the location folder. The location is determined to always be situated in the <data/processed> location as we should only train with datasets already processed.  
Label is pickled during training as well, as it takes the direct sub-folder names (real, fake) as the labels for each photos.  

## Testing

There are two ways you can test your model.

1. Launch a live cam and see if the screen is capturing you or your images as real or spoofed: `make predict_video model=vgg16_pretrained.model detector=face_RFB`. Similarly, you can change the used model and face_detector.
2. Classify photos using a pretrained model. This is similar to step 3 in "Preprocessing images based on unclassified images": `make classify location=attendance_photos`

------------

## Future Work

1. Allow more customization for testing - classify photos not just in bulk but solo as well. Allow classified photos to be saved elsewhere.  
2. The local api implementation is done in this github link: <https://github.com/jindongyang94/anti-spoofing_api>

------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
