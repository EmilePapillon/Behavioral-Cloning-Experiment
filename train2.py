# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4:

import cv2
import numpy as np
import tensorflow as tf
import csv
import os
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#returns a numpy array that is a copy of the input array with a new row of zeros at its end
def appendARowOfZeros(array):
	__array = np.zeros(array.shape[0], array.shape[1]+1) 
	__array[:,0:array.shape[1]] = array
	return __array

#augments and pre-processes (shuffle, crop)  the datai
#reformats the data : [img_path angle]
def preprocess(samples):
	#TODO calculate a more accurate correction factor
	correction_factor = 0.2
	
	#augment with left and right images
	samples_center = appendARowOfZeros(samples[:,np.array([0,3])]) #row of zeros is appended to indicate that this is not a flipped image
	samples_left = appendARowOfZeros(samples[:,np.array([1,3])])
	samples_left[:,1] += correction_factor #slightly change the steering angle label for the left camera
	samples_right = appendARowOfZeros(samples[:,np.array([2,3])])
	samples_right[:,1] -= correction_factor 
	samples_mirror = samples_center
	samples_mirror[:,1] *= -1 #flip the angle
	samples_mirror[:,-1] = 1 #this one is flipped	
	
	preprocessed_samples = np.append(samples_center, [ samples_left, samples_right, samples_mirror ],axis=0) 
	return shuffle(preprocessed_samples)


train_samples, validation_samples = train_test_split(lines, test_size= 0.2)
	
#TODO :make sure samples are shuffled somewhere in the process

def generator(samples, batch_size=10):
        
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
            		images = []
            		angles = []
			
			for batch_sample in batch_samples:
				name = '../data/IMG/'+batch_sample[0].split('/')[-1]
				image = cv2.imread(name)
				angle = float(batch_sample[1])
				if bool(batch_sample[2]): 
					image = np.fliplr(image) # if this is one of the inverted samples, we must now invert the image
		    		images.append(image)
		    		angles.append(angle)
			X_train = np.array(images)
			y_train = np.array(angles)
            		yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=5)
validation_generator = generator(validation_samples, batch_size=5)
ch, row, col = 3, 160, 320  # Trimmed image format

#model
model = Sequential()
model.add(Cropping2D(cropping = ((70,25),(0,0)),input_shape=(row,col,ch)))
model.add(Lambda(lambda x: (x-128)/128))
model.add(Convolution2D(16,3,3, 
	activation='relu', 
	border_mode= 'same'))
model.add(Convolution2D(16,3,3,
	activation='relu',
	border_mode='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(32,3,3, 
	activation='relu',
	border_mode='same'))
model.add(Convolution2D(32,3,3, 
	activation='relu',
	border_mode='same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

#train
model.compile(loss='mse', optimizer='adam')

#If the above code throw exceptions, try : 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=1, verbose = 1)

model.save('../model.h5')

'''
#This is to display training / validation losses
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
