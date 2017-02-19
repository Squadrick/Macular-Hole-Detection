from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
import keras
import cPickle as pickle

data = pickle.load(open('../data/images.p','rb'))

X_train = data['x_train']
Y_train = data['y_train']
X_test = data['x_test']
Y_test = data['y_test']

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(192,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(256))
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

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=8),
        samples_per_epoch=250, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, batch_size=16)

fileName = 'weights/' + str(score[1])[0:5] + '.h5'
model.save_weights(fileName)

print(score)

