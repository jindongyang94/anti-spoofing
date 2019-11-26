.PHONY:

classify:
	python neural/test.py classify_images --location=/Users/jindongyang/Documents/repos/hubble/hubble_projects/hubble_spoofing_detection/dataset/unsorted \
	--model=vgg16_pretrained.model --le=le.pickle --detector=face_detector

video:
	python neural/test.py video_demo --model=vgg16_pretrained.model --le=le.pickle --detector=face_detector