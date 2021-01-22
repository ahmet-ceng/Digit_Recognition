import tkinter as tk
from tkinter import *
import PIL
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

#Classes represent each digit
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Load and prepare the image and return the prediction for that image
def load_prepare_predict():
    #Loading our model
    model = tf.keras.models.load_model('model.h5')

    #Loading the image (image name is 'image.png' because we captured users drawing and saved is as 'image.png')
    img = cv2.imread('image.png', 0)
    img = cv2.bitwise_not(img)

    #Showing captured image to user. This was mainly for test but we didn't delete it because it looks good.
    cv2.imshow('img', img)

    #Resizing and reshaping to match images in mnist data
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)

    #Preparing pixels to normalize inputs
    img = img.astype('float32')
    img = img / 255.0

    #taking and returning the prediction
    pred = model.predict(img)
    return pred

#Allowing user to draw black ovals with width:40
def painting(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    #Width 40 is ideal because lower values causes image to be distorted and it reduces the accuracy.
    cv.create_oval(x1, y1, x2, y2, fill="black", width=40)
    draw.line([x1, y1, x2, y2], fill="black", width=40)


#Saves the image user drew, gets the prediction and inserts it to GUI
def btnPredict():
    #Saving the digit user drew as 'image.png'
    filename = "image.png"
    image1.save(filename)

    #Getting the prediction of the image
    pred = load_prepare_predict()

    #Printing the prediction and accuracy
    print('argmax', np.argmax(pred[0]), '\n', pred[0][np.argmax(pred[0])], '\n', classes[np.argmax(pred[0])])

    #Inserting the prediction and accuracy to GUI
    txt.insert(tk.INSERT, "{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])],
                                                     round(pred[0][np.argmax(pred[0])] * 100, 3)))

#Clears GUI screen
def btnClear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)



#Creating instance of TK
root = Tk()
#Disabling resizability.
root.resizable(0, 0)

#Creating a white canvas with width,height of 500. We also made the cursor: circle
cv = Canvas(root, width=500, height=500, bg='white', cursor='circle')
cv.pack(expand=YES, fill=BOTH)

#Creating an empty image and draw object to draw on. It is not visible.
image1 = PIL.Image.new("RGB", (500, 500), (255, 255, 255))
draw = ImageDraw.Draw(image1)

#Binding mouse 1 to painting function.
cv.bind("<B1-Motion>", painting)

#Creating the text window to display predictions and accuracy.
txt = tk.Text(root, bd=3, exportselection=0, bg='WHITE', font='Helvetica', padx=10, pady=10, height=5, width=20)

#Creating and binding predict and clear buttons.
btnPredict = Button(text="Predict", command=btnPredict, bg='green', fg='white')
btnClear = Button(text="Clear", command=btnClear, bg='red', fg='white')

#Packing buttons and text in order of predict>clear>txt area.
btnPredict.pack()
btnClear.pack()
txt.pack()

#Title of window
root.title('Digit Recognizer')

root.mainloop()


