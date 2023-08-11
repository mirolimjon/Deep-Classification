import tensorflow as tf
import numpy as np
import os

# ////////////////////////////////////////
# Load the data

labels = []
images = []

filepath = "C:/Users/User/Desktop/Classification/CNN/data/"

# Loop through
for i in os.listdir(filepath):
  for image in os.listdir(filepath + i):
    labels.append(i)
    images.append(filepath + i + "/" + image)  # Gets images location

# print(labels[:10])
# print(images[:10])
# ////////////////////////////////////////
# Getting data ready
unique_labels = np.unique(labels)


# /////////////////////////////////////////
# Turn data into boolean data types
boolean_labels = [
    label == unique_labels for label in labels
]

# print(boolean_labels[:10])

# //////////////////////////////////////////

# Split the data into train and validation data sets

X = images
y = boolean_labels

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# len(X_train), len(X_valid), len(y_train), len(y_valid)


# //////////////////////////////////////////////
# Processing images with tensorflow

# Setup image size
IMG_SIZE = 224

# Create a function for processing images
def processing_images(filepath):
  """
    Takes image filepath, load the data and return to processed image

    1. Read image with tf.io.read_file with `image` variable
    2. Turn image into tensors with 3 color channels
    3. Normalize image color channels
    4. Resize image
    5. Return processed image
  
  """
  image = tf.io.read_file(filepath)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

# print(processing_images(images[1]))

# During created data batches we need images label, we can pre-process images with  processing_images function, however
# We can not get their labels, so that let's create a function for get them

def get_images_label(filepath, labels):
  """
    Gets images filepath and labels, return to processed images and label pairly
  """
  image = processing_images(filepath)
  return image, labels

# print(get_images_label(images[1], labels[1]))

# /////////////////////////////////////////////
# Create data batches
# Create a function for create data batches

def create_data_batches(x, y=None, valid_set = False, test_set = False, batch_size=32):
  """
    Gets datasets and create data batches. It also can create batches for test set and validation datasets
    if our data is train set shuffle dataset before creating data batch
  """
  # Test set
  if test_set:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(processing_images).batch(32)
    return data_batch
  
  # Validation set
  elif valid_set:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                               tf.constant(y)))
    data_batch = data.map(get_images_label).batch(32)
  
  # Train set
  else:
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                               tf.constant(y)))
    data = data.shuffle(buffer_size=len(x))
    data_batch = data.map(get_images_label).batch(32)
  return data_batch

train_data = create_data_batches(X_train, y_train)
valid_data = create_data_batches(X_valid, y_valid, valid_set=True)


# /////////////////////////////////////////////////

# MODELING
# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # // None is batch size

# Setup output shape for model
OUTPUT_SHAPE = len(unique_labels)

# Get pre trained model url(Tensorflow Hub)
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"

import tensorflow_hub as hub

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url = MODEL_URL):
  print("Now, building model for you project: ")
  model = tf.keras.Sequential([
      hub.KerasLayer(model_url), # // Layers input
      tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation="softmax"), # //Layer for output
  ])

  # Compile the model
  model.compile(
      loss = tf.keras.losses.CategoricalCrossentropy(),
      optimizer = tf.keras.optimizers.Adam(),
      metrics = ["accuracy"]
  )

  # Build model
  model.build(INPUT_SHAPE)
  print("Model has been built successfully!")
  # Return the model
  return model



import datetime

# Setup folder
logs = "C:/Users/User/Desktop/Classification/CNN/logs"

# Create a function for build callbacks
def create_tensorboard_callback(dir_name):
  log_dir = dir_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )

  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience = 3,
)

def train_model():
  """
  Trains a given model and returns the trained model
  """
  # Create a model
  model = create_model()
  # Create callbacks
  tensorboard = create_tensorboard_callback(dir_name=logs)
  # Fit the model
  model.fit(x=train_data,
            epochs=100,
            validation_data=valid_data,
            callbacks = [tensorboard, early_stopping]
  )
  return model


model = train_model()

model.save("C:/Users/User/Desktop/Classification/CNN/model/model.h5")