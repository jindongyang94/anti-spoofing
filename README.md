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

Credits goes to https://www.pyimagesearch.com/author/adrian/

This library is simply to show how you can train your own simple face detection wtih anti-spoofing function:  

1. Make a video of yourself for about 20 -30 seconds and save it under the data/external/portrait_videos/real folder .  
2. Take a video of the video of yourself and save it under the data/external/portrait_videos/fake folder .  
3. Split the videos to frames or photos to create your dataset using function: ''' make data_video portrait_videos face_detector 8 0 ''' . The frames will be saved under src/data/processed folder.
4. Train the model using src/models/train.py .  
5. Test the model using src/models/predict.py .  

Examples of the function commands are pasted at the top of all scripts.  
Use the makeFile functions instead of interacting with the scripts yourself.

------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
