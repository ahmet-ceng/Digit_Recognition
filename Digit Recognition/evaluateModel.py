from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical


#Loading and preprocessing dataset.
def load_dataset():
    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape dataset to have a single channel
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # one hot encode target values
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


# run the test harness for evaluating a model
def main():
    #Loading the dataset
    X_train, y_train, X_test, y_test = load_dataset()

    #Normalizing images by processing pixel data
    X_train, X_test = scale_pixels(X_train, X_test)

    #Loading model
    model = load_model('model.h5')

    #Evaluate our model and print accuracy
    loss, accuracy = model.evaluate(X_test, y_test)
    print("\nAccuracy of 'model.h5' is :", accuracy)
    print("Loss of 'model.h5' is :", loss)


main()
