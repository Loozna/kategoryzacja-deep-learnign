import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

img = ImageDataGenerator(rescale = 1.0/255.0,vertical_flip = True)

train_generator = img.flow_from_directory('Covid19-dataset/train',target_size = (256,256),color_mode = 'grayscale',batch_size = 32,class_mode = 'categorical')

test_generator = img.flow_from_directory('Covid19-dataset/test',target_size = (256,256),color_mode = 'grayscale',batch_size = 32,class_mode = 'categorical')



model = Sequential()
model.add(tf.keras.layers.Input(shape = (256,256,1)))
model.add(tf.keras.layers.Conv2D(2,4,strides = 2, activation = 'relu',padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = 2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8, activation = 'relu'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))
model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate = .001), loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])
print(model.summary())
history = model.fit(train_generator,epochs = 5)
print(model.evaluate(test_generator))
labels_predicted = np.argmax(model.predict(test_generator),axis = 1)
print(labels_predicted)
print(test_generator.labels)
print(classification_report(test_generator.labels,labels_predicted))



