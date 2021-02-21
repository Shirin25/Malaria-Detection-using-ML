import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib
import pickle,os
from keras.preprocessing.image import load_img
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app=Flask(_name_)



filename = 'finalized_model.sav'

model = pickle.load(open(filename, 'rb'))


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    img_path = request.files.get('image')
    img = image.load_img(img_path, target_size=(50,50)) 

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model_predict(img)
    pred = np.argmax(preds,axis = 1)
    return pred    


@app.route('/predict', methods=['GET', 'POST'])
def model_predict():
    print('PREDICT')
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(_file_)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        print(file_path)
        
        img = image.load_img(img_path, target_size=(50,50))
        
         #Preprocessing the image
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0) 
        
        
       
   
         pred = model_predict(img)
         pred = np.argmax(preds,axis = 1)
        
        
        pred = model.predict(img) #img
        os.remove(file_path)

        str1 = 'Malaria Parasitized'
        str2 = 'Normal'
        
        pred=[]
        if pred[0] == 0:
            s=str1
        else:
            s=str2
    
    return render_template('result3.html',result=s)

   
if _name_ == '_main_':
        app.run()