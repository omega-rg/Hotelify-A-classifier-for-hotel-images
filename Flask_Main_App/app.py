import sys
import os
import glob
import re
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "D:\\Real Estate CV\\Image Classifier\\Checkpoints\\model_data_augmennt_full_train_checkpoints"

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary

print('Model loading...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
model=load_model(MODEL_PATH)
#graph = tf.get_default_graph()

print('Model loaded. Started serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def pred_to_string(d):
    ans=""
    for classx,pred in d.items():
        ans+=classx+": "+str(pred)+", "
    return ans


preprocessing_function= lambda x: preprocess_input(x,mode='tf')

def preprocess_image(img_path):
    img=load_img(img_path,target_size=(256,256))
    img=img_to_array(img)
    img=preprocessing_function(img)
    return img

CLASS_DICT = {0: 'Balcony',
                1: 'Bar',
                2: 'Bathroom',
                3: 'Bedroom',
                4: 'Bussiness Centre',
                5: 'Dining room',
                6: 'Exterior',
                7: 'Gym',
                8: 'Living room',
                9: 'Lobby',
                10: 'Patio',
                11: 'Pool',
                12: 'Restaurant',
                13: 'Sauna',
                14: 'Spa'
            }

def model_predict(img_path, model):
    # original = image.load_img(img_path, target_size=(224, 224))

    # # Preprocessing the image
    
    # # Convert the PIL image to a numpy array
    # # IN PIL - image is in (width, height, channel)
    # # In Numpy - image is in (height, width, channel)
    # numpy_image = image.img_to_array(original)

    # # Convert the image / images into batch format
    # # expand_dims will add an extra dimension to the data at a particular axis
    # # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # # Thus we add the extra dimension to the axis 0.
    # image_batch = np.expand_dims(numpy_image, axis=0)

    # print('PIL image size = ', original.size)
    # print('NumPy image size = ', numpy_image.shape)
    # print('Batch image  size = ', image_batch.shape)

    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # processed_image = preprocess_input(image_batch, mode='caffe')
    
    # #with graph.as_default():    
        
    # preds = model.predict(processed_image)

    img=preprocess_image(img_path)
    preds=list(model.predict(np.expand_dims(img,0),batch_size=1).flatten())

    print("**********************************************************")
    print("**********************************************************")
    print("**********************************************************")
    print("**********************************************************")
    print("**********************************************************")
    print("**********************************************************")

    d={}
    for i,prob in enumerate(preds):
        d[CLASS_DICT[i]]=prob
#     {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    d={classx: prob for classx,prob in sorted(d.items(),key=lambda item:item[1],reverse=True)[:3]}
    
    print('Deleting File at Path: ' + img_path)

    os.remove(img_path)

    print('Deleting File at Path - Success - ')

    return pred_to_string(d)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('Begin Model Prediction...')

        # Make prediction
        result = model_predict(file_path, model)

        print('End Model Prediction...')

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        
        return result
    return None

if __name__ == '__main__':    
    app.run(debug=False, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()