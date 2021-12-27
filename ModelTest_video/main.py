from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import cv2
import time

def visualize(**images):
    """
    PLot images in one row.
    """
    fig = plt.figure(figsize=(8, 8))
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    ax1 = plt.subplot(gs1[0])
    ax1.imshow([[0, 1], [2, 1]])

    ax1.axis('off')
    poss = {
        0: pred[0][0] * 100,
        1: pred[0][1] * 100,
        2: pred[0][2] * 100,
        3: pred[0][3] * 100,
        4: pred[0][4] * 100
    }
    plt.title('Result : ' + labels[np.argmax(prediction)],bbox={"facecolor":"blue", "alpha":0.5})
    s = "Flood : " + str(poss[0]) + "\n" + "General : " + str(poss[1]) + "\n" + "Fire : " + str(
        poss[2]) + "\n" + "Post-Fire : " + str(poss[3]) + "\n" + "Building Disaster: " + str(poss[4]) + "\n"
    ax1.text(0.5, -0.5, s, size=13, ha="center",
             transform=ax1.transAxes,bbox={"facecolor":"orange", "alpha":0.5})

    plt.imshow(image)
    plt.show()

labels = {
0: 'Flood',
1:'General',
2:'Fire',
3 :'Post - Fire',
4 :'Building Disaster'}
# Load the model

model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# Read the video I guess

#added the timer
start = time.time()

vidcap = cv2.VideoCapture('Fire.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file

  image = Image.open('frame%d.jpg' % count)
  # resize the image to a 224x224 with the same strategy as in TM2:
  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  # turn the image into a numpy array
  image_array = np.asarray(image)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  data[0] = normalized_image_array

  # run the inference
  prediction = model.predict(data)

  print(labels[np.argmax(prediction)])

  pred = np.array(prediction)

  visualize(
      image=image,
  )

  success,image = vidcap.read()
  os.remove('frame%d.jpg' % count)
  count += 1

stop = time.time()

print('Time: ', stop - start)

