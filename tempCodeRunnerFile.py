from flask import Flask, render_template, Response, request, session , redirect
# from transformers import BartForConditionalGeneration, BartTokenizer
import time
import os


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
import threading




import pathlib
data_dir = "leaf_photos"
data_dir = pathlib.Path(data_dir)


batch_size = 32
img_height = 180
img_width = 180



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



class_names = train_ds.class_names
print(class_names)



num_classes = 16

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])



model.load_weights('my_checkpoint.ckpt')

app = Flask(__name__)

app.secret_key = "dfjklwcnhe45ui672cvn894726c46m23w7845cvyt2u34rv7bfwuhjdf"


def generate_result(file_name):
  

    test_image_path = 'static/uploads/'+file_name  

    img = keras.preprocessing.image.load_img(
        test_image_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    #PIL.Image.open(test_image_path)

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))


def getFileName():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = timestr+".JPG"
    # timestr ="asset"
    path = os.path.join(os.getcwd()+'\\static\\uploads\\', (timestr+".JPG"))
    return path, file_name

@app.route("/", methods = ['GET','POST'])
def index():
    return render_template("index.html")



@app.route("/upload_file", methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
      f = request.files['file']
      file_path, file_name  = getFileName() 
      print(file_path)
      session['file_name'] = file_name
      f.save(file_path)
      f.close()
      print("file saved")
      return "<script>alert('File upload successful.'); window.open('/preview_file','_self')</script>"
    
    return render_template("upload-file.html")

@app.route("/preview_file", methods = ['GET','POST'])
def preview_file():
    file_name =  session['file_name']

    if file_name is None:
        return redirect('/')
    
    file_path =  'static/uploads/'+file_name  
    if request.method == 'GET':
      
        return render_template("preview-file.html", preview_img=file_path, result_txt=None )
    else:
        
        result_summary = generate_result(file_name)
        return render_template("preview-file.html", preview_img=file_path, result_txt=result_summary )





if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0")