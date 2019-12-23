import os

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


class Model:
	def __init__(self, width, height, depth, classes):
		self.inputShape = (height, width, depth)
		self.chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			self.inputShape = (depth, height, width)
			self.chanDim = 1

		self.classes = classes

	def build_liveness(self):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(16, (3, 3), padding="same",
							input_shape=self.inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=self.chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=self.chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=self.chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=self.chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(self.classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

	def build_VGG(self):
		# build the VGG16 network
		model = Sequential()

		model.add(ZeroPadding2D((1, 1), input_shape=self.inputShape))
		model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		weights_path = os.path.join('neural', 'pretrained_weights', 'vgg16_weights.h5')
		model.load_weights(weights_path, by_name=True)

		top_model = Sequential()
		top_model.add(Flatten(input_shape=model.output_shape[1:]))
		top_model.add(Dense(256, activation='relu'))
		top_model.add(Dropout(0.5))
		top_model.add(Dense(self.classes, activation='sigmoid'))

		# add the model on top of the convolutional base
		model.add(top_model)

		return model
