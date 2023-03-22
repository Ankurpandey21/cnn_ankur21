import numpy as np
from keras.preprocessing import image
import cv2
import time
import os
import tensorflow as tf
from keras.utils import to_categorical
import cv2
import glob
import random
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
import keras

classes = ['Apple','Banana','Pear','Lychee','Brinjal','Tomato','Cauliflower','coriander','cabcicam' ]

#step1 Initializing CNN
classifier = Sequential()

# step2 adding 1st Convolution layer and Pooling layer
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step3 adding 2nd convolution layer and polling layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


#step4 Flattening the layers
classifier.add(Flatten())

#step5 Full_Connection

classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=9,activation = 'softmax'))

#step6 Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print("Training cnn")


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:\\New folder\\srai-master\\python cnn\\train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\New folder\\srai-master\\python cnn\\test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
print("Image Processing.......Completed")

a = glob.glob('weight.h5')
if (os.path.exists("C:\\New folder\\srai-master\\weight.h5")):
    classifier.load_weights('C:\\New folder\\srai-master\\weight.h5')
    print("weight found .. loading weight.")
else:
    classifier.fit(training_set, validation_data=test_set, batch_size=32, epochs=20 )
    classifier.save_weights('weight.h5')
    print("weight not found ... creating one...")
print(training_set.class_indices)

vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\New folder\\srai-master\\python cnn\\result\\final'+str(i)+".jpg", frame)
    test_image = keras.utils.load_img('C:\\New folder\\srai-master\\python cnn\\result\\final'+str(i)+".jpg", target_size = (64, 64))
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    result1=result[0]
    for j in range(9):
        
        if result1[j] == 1:
            break
    print(classes[j])  

    os.remove('C:\\New folder\\srai-master\\python cnn\\result\\final'+str(i)+".jpg")
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
# C:\New folder\srai-master\python cnn\result\final.jpg