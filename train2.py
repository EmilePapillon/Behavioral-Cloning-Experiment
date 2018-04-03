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
#with open('../data/driving_log.csv', 'r', newline='') as csvfile:
with open('../data/driving_log.csv') as csvfile:
	#dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=',')
	#csvfile.seek(0) #rewind
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size= 0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            #TODO: trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)
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

"""
#former code :
=======
>>>>>>> e3a675f8b754e4a6231648abfcb517f2820b82c1
=======
>>>>>>> e3a675f8b754e4a6231648abfcb517f2820b82c1
model.fit_generator(train_generator, 
	samples_per_epoch= len(train_samples),
	validation_data = validation_generator,
	nb_val_samples=len(validation_samples),
	nb_epoch = 1, verbose = 1 )
<<<<<<< HEAD
<<<<<<< HEAD
"""

model.save('model.h5')

'''
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
