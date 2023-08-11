import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np


labels = []
images = []

filepath = "C:/Users/User/Desktop/Classification/CNN/data/"

# Loop through
for i in os.listdir(filepath):
  for image in os.listdir(filepath + i):
    labels.append(i)
    images.append(filepath + i + "/" + image)  # Gets images location
unique_labels = np.unique(labels)


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
  image = np.expand_dims(image, axis=0)
  return image

# # Load model

path = "C:/Users/User/Desktop/Classification/CNN/test/jyh_1.jpg"
test_img = processing_images(path)


def load_model(model_path):
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

model = load_model("C:/Users/User/Desktop/Classification/CNN/model/20230517-03051684295165-MobileNetV2.h5")
# C:\Users\User\Desktop\Classification\CNN\model\


prediction = model.predict(test_img)
print("Similarity: {:.2f} %".format(np.max(prediction)*100))
print(f"Max value (Probability prediction) {np.max(prediction)}")
print(f"Sum: {np.sum(prediction)}")
print(f"Max. index: {np.argmax(prediction)}")
print(f"Label: {unique_labels[np.argmax(prediction)]}")



# Turn prediction probabilities into their respective labels
# def get_pred_label(prediction_probabilities):
#   """
#     Turns on array of prediction probabilities into a label
#   """
#   return unique_labels[np.argmax(prediction_probabilities)]

# print(get_pred_label(prediction))