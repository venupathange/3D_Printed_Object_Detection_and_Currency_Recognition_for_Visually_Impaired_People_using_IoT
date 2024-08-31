import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array , load_img
import h5py
import keras
import cv2

from tensorflow.keras.models import load_model
from keras.models import load_model
from imutils.video import VideoStream
# Create BlynkTimer Instance
import BlynkLib

#GPIO.setmode(GPIO.BOARD)
GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)

model_save = tf.keras.models.load_model("")

#img_file = 'Validation/fire/83.jpg'
def img_processing():
    camera_img()
    img='frame_test.jpg'
    image = load_img(img,target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = image / 255
    # AI Model predict
    res = model_save.predict(image)
    
    pred = np.argmax(res, axis=1)
    print(pred)
    print(pred)
    if pred[0] == 0:
        print('Fire')
            
    elif pred[0] == 1:
        print('NO Fire')
        
    else:
        print('Unknown')
  

        
def camera_img():
    # define a video capture object
    vid = cv2.VideoCapture(0)
  
    ret, frame = vid.read()

    cv2.imwrite('frame_test.jpg', frame)
  
    # After the loop release the cap object
    vid.release()

    
    
