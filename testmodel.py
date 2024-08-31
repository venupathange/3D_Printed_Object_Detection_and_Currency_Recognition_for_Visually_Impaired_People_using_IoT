import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array , load_img
import h5py
import keras


from tensorflow.keras.models import load_model
from keras.models import load_model
model_saved = tf.keras.models.load_model("model_Classifier.h5")
import cv2

#base_path = 'Datasets'
def Camera():

    vid = cv2.VideoCapture(0)

    ret, frame = vid.read()

    cv2.imwrite('frame_test.jpg', frame)

    # After the loop release the cap object
    vid.release()



def Get_Result():
    img_file = 'frame_test.jpg'
    image = load_img(img_file,target_size=(224,224))
    plt.imshow(image)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = image / 255
    res = model_save.predict(image)
    print("Predicted Result",res)
    print("Result",res[0])
    # print("Sorted list",np.sort(res))
    # print("min",min(res[0]),'\n MAX',max(res[0]))
    # print("minimum value", np.where(res[0] == min(res[0])))
    # print("maximum value",np.where(res[0] == max(res[0])))

    pred = np.argmax(res, axis=1)
    res = model_save.predict(image)
    print(res)
    pred = np.argmax(res, axis=1)
    print(pred)
    print(pred)
    if pred[0] == 0:
        print('100')
    elif pred[0] == 1:
        print('200')
    elif pred[0] == 2:
        print('2000')
    elif pred[0] == 3:
        print('500')
    elif pred[0] == 4:
        print('50')
    elif pred[0] == 5:
        print('10')
    elif pred[0] == 6:
        print('20')
    else:
        print('not indian cuurency')
    

while True:
    Camera()
    #delay(1000)
    Get_Result()