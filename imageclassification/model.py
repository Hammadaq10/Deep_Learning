from keras.datasets import cifar10 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

