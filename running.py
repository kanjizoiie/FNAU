import time
import model
import platform
import tensorflow as tf
import cv2
import numpy
import pandas

processing_time_list = []
time_list = []

sess = tf.Session(config = tf.ConfigProto(log_device_placement=True))
shutdown = False

m = model.food_model()
m.create()
m.compile()
m.load("25_epochs.h5")

vc = cv2.VideoCapture()
vc.open("http://192.168.0.178:8080/video")

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

p = platform.system().lower()
print(m.classes())
inv_classes = dict(map(reversed, m.classes().items()))

while not (shutdown):
    # Special function for windows because python does not have high precision timer in this version (3.6.6)
    if p == "windows":
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

        print("Prediction: ", pred)
        print("Prediction class: ", inv_classes[int(numpy.rint(pred))])

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

        print("Prediction: ", pred)
        print("Prediction class: ", inv_classes[int(numpy.rint(pred))])

    processing_time_elapsed = end_processing_time - start_processing_time
    time_elapsed = end_time - start_time
    processing_time_list.append(processing_time_elapsed)
    time_list.append(time_elapsed)
vc.release()
if p == "windows":
    cv2.destroyWindow("preview")

average_processing_time = numpy.average(processing_time_list)
average_time = numpy.average(time_list)


stdd_processing_time = numpy.std(processing_time_list)
stdd_time = numpy.std(time_list)

print("Average Processing time: ", average_processing_time, " ms")
print("STD deviation processing time: ",  stdd_processing_time, " ms")
print("Average time: ", average_time, " ms")
print("STD deviation time: ",  stdd_time, " ms")

df = pandas.DataFrame(data={"processing_time": processing_time_list, "time": time_list})
df.to_csv("./data.csv", sep=',',index=False)