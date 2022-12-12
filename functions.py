import cv2
import dlib
from deepface import DeepFace

def face_detection_yunet(image):
    import cv2
    import dlib
    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (0,0))
    h, w, _ = image.shape
    face_detector.setInputSize((w,h))
    _, faces = face_detector.detect(image)
    return [dlib.rectangle(face[0], int(face[1] * 1.15), int((face[0]+face[2]) * 1.05), face[1]+face[3]) for face in faces]

def face_clean(image):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    shape_predictor = dlib.shape_predictor("algorithm/shape_predictor_68_face_landmarks.dat")
    face_box = face_detection_yunet(image)[0]
    shape = shape_predictor(image, face_box)
    face_chip = dlib.get_face_chip(image, shape)   
    return face_chip

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 1:
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
elif tf_version == 2:
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

def loadModel():

	num_classes = 7

	model = Sequential()

	#1st convolution layer
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	#fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))

	#---------------------------

	model.load_weights('./facial_expression_model_weights.h5')

	return model