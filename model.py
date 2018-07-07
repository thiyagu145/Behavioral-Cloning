import csv
import cv2
import numpy as np

lines=[]
with open ('/home/thiyagu_1405/carnd/CarND-Term1-Starter-Kit/model-2/Data/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                choice=np.random.choice(3)
                name = '/home/thiyagu_1405/carnd/CarND-Term1-Starter-Kit/model-2/Data/data/IMG/'+batch_sample[choice].split('/')[-1]
                image = cv2.imread(name)
                if choice==0:
                    angle = float(batch_sample[3])
                elif choice==1:
                    angle = float(batch_sample[3])+0.2
                else:
                    angle = float(batch_sample[3])-0.2
                ##Pre Processing
                ##Converting the color space
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ##Cropping the image
                image=image[50:140,:,:]
                ##Resizing the image
                image=cv2.resize(image, (200,66), interpolation = cv2.INTER_AREA)
                ##Flipping the images in random
                if random.uniform(0,1)<0.5:
                    image = np.fliplr(image)
                    angle = -angle
                images.append(image)
                angles.append(angle)
            
            
            # trim image to only see section with road
            X_train = np.array(images)/127.5-1.0
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.backend import tf as ktf

model=Sequential()
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu',input_shape=(66,200,3),subsample=(2,2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu',subsample=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu',subsample=(2,2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('final_model.h5')

