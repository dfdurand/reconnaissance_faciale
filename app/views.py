import cv2
import os
import werkzeug
import jsonpickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from app.faces_recognition import faceRecognitionPipeline
from flask import render_template, request, Response, jsonify

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        #save image in upload directory
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        #get predictions

        pred_image, prediction = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
        
        print("-----image prédire avec succès par le modèle-----")

    return render_template('gender.html', fileupload=True) #POST request


def names():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    filename = 'remote_pred_image.jpg'
    cv2.imwrite(f'./static/predict/{filename}', img)

    chemin = './static/predict/remote_pred_image.jpg' 

    pred_img, predictions = faceRecognitionPipeline(chemin) 

    reponse = predictions[-1]['prediction_name']

    print("-----image prédire avec succès par le modèle-----")
    print("**** le nom est : ", reponse)

    cv2.imwrite(f'./static/predict/{filename}', pred_img)

    return jsonify(reponse)


def test():

    back =  "salut ici le serveur "

    return jsonify(back)


def receive():

    imagefile = request.files['file']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    return "Image Uploaded Successfully"
