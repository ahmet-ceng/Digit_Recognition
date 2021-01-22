from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD



#Loading and preprocessing dataset.
def load_dataset():
    #Loading mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Reshaping the data
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    #Applying One-Hot Code technique to convert values into numerical form.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


#Preparing Pixel Data by normalizing inputs from 0–255 to 0–1
def scale_pixels(train, test):
    #Convert integers to float
    newTrain = train.astype('float32')
    newTest = test.astype('float32')

    #Normalize it to range 0-1
    newTrain = newTrain / 255.0
    newTest = newTest / 255.0

    #Return normalized images
    return newTrain, newTest


#Defining a baseline Convolutional Neural Network (CNN) model.
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    #Defining optimizer and Compiling the model
    optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    #Loading the dataset
    X_train, y_train, X_test, y_test = load_dataset()

    #Normalizing images by processing pixel data
    X_train, X_test = scale_pixels(X_train, X_test)

    #Loading model
    model = create_model()

    #Fitting X_train and y_train into our model
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    #Saving the model as 'model.h5'
    model.save('model.h5')


main()

