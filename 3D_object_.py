
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import shutil


TRAINING_DIR = "New data/Train"

training_datagen = ImageDataGenerator(rescale = 1./255,
                                      horizontal_flip=True, rotation_range=10,height_shift_range=0.2,
                                      fill_mode='nearest')

VALIDATION_DIR = "New data/Test"
validation_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True, rotation_range=10,height_shift_range=0.2,
                                      fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=(224,224),class_mode='categorical',
  batch_size = 4
)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(224,224),class_mode='categorical',
  batch_size= 2
)


# In[3]:


learning_rate=0.001


# In[4]:


from tensorflow.keras.optimizers import RMSprop,Adam

model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(224, 224, 3)),
          tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
          tf.keras.layers.Conv2D(256, (5,5), activation='relu'),
          tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
          tf.keras.layers.Conv2D(384, (5,5), activation='relu'),
          tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(2048, activation='relu'),
          tf.keras.layers.Dropout(0.25),
          tf.keras.layers.Dense(1024, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss='categorical_crossentropy',metrics=['acc'])
#model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['acc'])
model.summary()


# In[17]:


# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('val_acc')>=0.98):
#       print('\nReached ^98%')
#       self.model.stop_training = True
# callbacks = myCallback()

history = model.fit(
    train_generator,
    steps_per_epoch = 15,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 30
    #callbacks=[callbacks]
)


# In[22]:


model.save('model_3d.h5',include_optimizer="False")


# In[24]:


model.save('my_model_3D.keras')#keras.saving.save_model(model, 'my_model.keras')`. 

