import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import jsonify

from _caffe import lenet
from caffe import imagenet
import numpy as np
from itertools import chain

UPLOAD_FOLDER = os.path.dirname(__file__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/lenet', methods=['GET', 'POST'])
def predict_lenet():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = 'image' + os.path.splitext(file.filename)[1] #secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Make prediction
            prediction = net_lenet.predict(filepath)
            prediction = np.asarray(prediction).flatten()
            pred_classes = np.argsort(np.asarray(prediction).flatten())[::-1]
            prediction = np.sort(prediction)[::-1]

            return jsonify(best=pred_classes[0], \
                probabilities=prediction.tolist(), predictions=pred_classes.tolist())
        else:
            if not allowed_file(file.filename):
                return jsonify(error="Invalid extension")

    # Show form to upload image
    return '''
    <!doctype html>
    <title>Upload image file</title>
    <h1>Upload image</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=image>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/imagenet', methods=['GET', 'POST'])
def predict_imagenet():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = 'image' + os.path.splitext(file.filename)[1] #secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Make prediction
            prediction = net_imagenet.predict(filepath)
            prediction = np.asarray(prediction).flatten()
            pred_classes = np.argsort(np.asarray(prediction).flatten())[::-1]
            prediction = np.sort(prediction)[::-1]

            return jsonify(best=pred_classes[0], \
                probabilities=prediction.tolist(), predictions=pred_classes.tolist())
        else:
            if not allowed_file(file.filename):
                return jsonify(error="Invalid extension")

    # Show form to upload image
    return '''
    <!doctype html>
    <title>Upload image file</title>
    <h1>Upload image</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=image>
         <input type=submit value=Upload>
    </form>
    '''
    
if __name__ == '__main__':
    ''' We provide access to two trained models:
        1. Lenet trained with MNISTClassifier for handwritten digit recognition
        2. Imagenet for object recognition 
        Both models are loaded before the web server. '''

    # Set the path to your trained models
    trained_models_path = '/home/allan/Git/caffe/examples'

    net_lenet = lenet.MNISTClassifier(os.path.join(trained_models_path, 'lenet/lenet.prototxt'), \
                                        os.path.join(trained_models_path, 'lenet/lenet_iter_10000'))
    net_lenet.caffenet.set_phase_test()
    net_lenet.caffenet.set_mode_gpu()


    net_imagenet = imagenet.ImageNetClassifier(os.path.join(trained_models_path, 'imagenet/imagenet_deploy.prototxt'), \
                                        os.path.join(trained_models_path, 'imagenet/caffe_reference_imagenet_model'))
    net_imagenet.caffenet.set_phase_test()
    net_imagenet.caffenet.set_mode_gpu()
    
    app.debug = False
    app.run(host='0.0.0.0')