import cv2
import numpy as np
import tensorflow as tf
import csv
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda, Input
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../data/driving_log.csv', 'r', newline='') as csvfile:
	reader = csv.reader(csvfile)
	#has_header = csv.Sniffer().has_header(csvfile.read(1024))
	#csvfile.seek(0) #rewind
	#if has_header:
	#	next(reader)
	for line in reader:
		lines.append(line)

images= []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		images.append(np.fliplr(image))
		measurement = float(line[3])
		measurements.append(measurement)
		measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)
#print(X_train.shape)
#print(y_train.shape)
#model

inpt = Input(shape = (160,320,3))
normalize= (Lambda(lambda x: (x-128)/128))(inpt)
crop = Cropping2D(cropping=((70,25),(0,0)))(normalize))
conv1= Convolution2D(32,3,3, activation='relu', border_mode='same')(crop)
conv2=Convolution2D(32,3,3, activation='relu', border_mode='same')(conv1)
maxpool1= MaxPooling2D()(conv2)
conv3= Convolution2D(64,3,3, activation='relu', border_mode='same')(maxpool1)
conv4= Convolution2D(64,3,3, activation='relu', border_mode='same')(maxpool2))
maxpool2= MaxPooling2D(conv4)
fc1 = Dense(120)(maxpool2)
fc2= Dense(84)(fc1)
fc3= Dense(1)(fc2)
model = Model(inputs= inpt, outputs= fc3)

#train
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.3, shuffle=True, nb_epoch = 5 )

model.save('model.h5')

