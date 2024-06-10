from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Dropout

#deepfake logic to add convolutional layer, pooling layer using either max/min/avg layer along with either with or without dropouts
model=Sequential()
model.add(Conv2D(32, (3,3),input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.1))
model.add(Flatten())

#deepfake logic for hidden layers with activation function relu and can have manipulation of different number of kernels
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#deepfake logic to compile the whole model to have a particular optimizer either sgd/adagrad/adadelta/adam with loss function for binary class classifciation which is binar_crossentropy and metric for accuracy
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

#to preprocess the images, the module from keras is imported and all the parameters are set
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#flow_from_directory is used to get the different classes of training and testing folder from the machine
training_set=train_datagen.flow_from_directory('./train', target_size=(64,64), batch_size=8, class_mode='binary')
test_set=test_datagen.flow_from_directory('./test', target_size=(64,64), batch_size=8, class_mode='binary')

#after preprocessing and compiling the model, fitting the model accordingly using epoch
model.fit(training_set, steps_per_epoch=20, epochs=15)

#this will print the test accuracy and losses
test_loss, test_acc=model.evaluate(test_set, verbose=2)

print("\n\n")
print("Tesy Loss: \t", test_loss, '\n')
print("Test Accuracy: \t", test_acc, '\n')