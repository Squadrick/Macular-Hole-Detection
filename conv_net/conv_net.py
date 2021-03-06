from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
import keras
import cPickle as pickle
from datetime import datetime

data = pickle.load(open('../data/images.pkl','rb'))

X_train = data['x_train']
Y_train = data['y_train']
X_val   = data['x_val']
Y_val   = data['y_val']
X_test  = data['x_test']
Y_test  = data['y_test']

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(384,512,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

opti = Nadam()

datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

model.compile(loss='binary_crossentropy',optimizer=opti,metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, 
                                 Y_train, 
                                 batch_size=4, 
                                 shuffle=True),
                    samples_per_epoch=1000, 
                    nb_epoch=2, 
                    verbose=1, 
                    validation_data=datagen.flow(X_val,
                                                 Y_val, 
                                                 batch_size=4,
                                                 shuffle=True),
                    nb_val_samples=100)

#score = model.evaluate(X_test, Y_test, batch_size=16)
score = model.evaluate_generator(datagen.flow(X_test, 
                                              Y_test, 
                                              batch_size=4), 
                                              val_samples=100)


fileName = 'weights/' + str(score[1])[0:5] +', ' + str(datetime.now()) +  '.h5'
model.save_weights(fileName)

print(score)
