import cv2
import numpy as np
import glob
import cPickle as pickle

size = (256, 192)
pos_filepath = glob.glob("../data/positive/*.jpg")
neg_filepath = glob.glob("../data/negative/*.jpg")

x_pos = np.array([np.array(cv2.resize(cv2.imread(fname), size)) for fname in pos_filepath])
x_neg = np.array([np.array(cv2.resize(cv2.imread(fname), size)) for fname in neg_filepath])

y_pos = np.ones(x_pos.shape[0])
y_neg = np.zeros(x_neg.shape[0])

x_pos_train = x_pos[:65]
y_pos_train = y_pos[:65]
x_pos_test  = x_pos[65:]
y_pos_test  = y_pos[65:]

x_neg_train = x_neg[:140]
y_neg_train = y_neg[:140]
x_neg_test  = x_neg[140:]
y_neg_test  = y_neg[140:]

x_train = np.append(x_pos_train, x_neg_train, axis = 0)
y_train = np.append(y_pos_train, y_neg_train, axis = 0)

x_test = np.append(x_pos_test, x_neg_test, axis = 0)
y_test = np.append(y_pos_test, y_neg_test, axis = 0)

data = {'x_train': x_train,
        'y_train': y_train,
        'x_test' : x_test,
        'y_test' : y_test}

pickle.dump(data, open('../data/images.p', 'wb'), protocol=2)
