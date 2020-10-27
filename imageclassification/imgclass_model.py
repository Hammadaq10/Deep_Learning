from keras.datasets import cifar10 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# convert to float type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize the testing features
x_train = x_train/255.0
x_test = x_test/255.0

# one-hot encoding on labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

classes = y_test.shape[1]

# create a convolutional sequential model 
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
        padding='same', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(classes,activation='softmax'))

sgd = SGD(learning_rate=0.01,momentum=0.9, decay=(0.01/25),nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=10, batch_size=32)

accuracy = model.evaluate(x_test,y_test)
print(accuracy*100)
model.save("imgclassification_model_cifar10.h5")
