import cv2
import numpy as np
import tensorflow as tf
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
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
	for i in range(3)
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
#print(X_train.shape)
#print(y_train.shape)
#model

#TODO find a way to make a non-sequential model like the ResNets
model = Sequential()
model.add(Lambda(lambda x: (x-128)/128, input_shape = (160,320,3)))
model.add(Convolution2D(3,1,1, activation='relu'))
model.add(Convolution2D(10,3,3, activation='relu', border_mode='same'))
#model.add(MaxPooling2D())
model.add(Convolution2D(10,3,3, activation='relu', border_mode='same'))
#model.add(MaxPooling2D())
model.add(MaxPooling2D())
model.add(Convolution2D(10,3,3, activation='relu', border_mode='same'))
model.add(Convolution2D(10,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

#train
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.3, shuffle=True, nb_epoch = 5 )

model.save('model.h5')

