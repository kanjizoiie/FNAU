import time
import model
import platform
import tensorflow as tf
import cv2
import numpy

sess = tf.Session(config = tf.ConfigProto(log_device_placement=True))
shutdown = False

m = model.food_model()
m.create()
m.compile()
m.load("25_epochs.h5")

vc = cv2.VideoCapture()
vc.open("http://192.168.0.100:8080/video")

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
print(m.classes())
inv_classes = dict(map(reversed, m.classes().items()))

# Special function for windows because python does not have high precision timer in this version (3.6.6)
if platform.system().lower() == "windows":
    while not (shutdown):

        # Read the data.
        rval, frame = vc.read()

        # Time the pre processing
        start_processing_time = time.clock()
        img_resize = cv2.resize(frame, (64, 64))
        img_reshaped = img_resize.reshape([1, 64, 64, 3])
        img_divided = img_reshaped / 255
        end_processing_time = time.clock()

        # Time the prediction
        start_time = time.clock()
        # TODO: implement prediction function here [X]
        pred = m.predict(img_divided)
        end_time = time.clock()


        print(pred)
        print(inv_classes[int(numpy.rint(pred))])


        # Show the image captured from the webcam
        cv2.imshow("preview", frame)
        # Wait for a keypress.
        key = cv2.waitKey(20)
        # If ESC key is pressed exit the program.
        if key == 27:  # exit on ESC
            shutdown = False
            break


# The normal way to implement a timing function on other platforms.
else:
    while not(shutdown):
        while not (shutdown):

            # Read the data.
            rval, frame = vc.read()

            # Time the pre processing
            start_processing_time = time.time()
            img_resize = cv2.resize(frame, (64, 64))
            img_reshaped = img_resize.reshape([1, 64, 64, 3])
            img_divided = img_reshaped / 255
            end_processing_time = time.time()

            # Time the prediction
            start_time = time.time()
            # TODO: implement prediction function here [X]
            pred = m.predict(img_divided)
            end_time = time.time()

            print(pred)
            print(inv_classes[int(numpy.rint(pred))])


vc.release()
if platform.system().lower() == "windows":
    cv2.destroyWindow("preview")