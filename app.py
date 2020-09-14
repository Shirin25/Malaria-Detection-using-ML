from flask import Flask,render_template,redirect,request,session,url_for,flash
import ibm_db
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import pickle
import numpy as np

from numpy import loadtxt
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.models import load_model
from flask import Flask, render_template, request
from keras.applications import imagenet_utils
import keras.models
import re
import sys 
import os
import base64
#import cv2
from io import BytesIO
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array
from base64 import b64encode
from keras.preprocessing.image import ImageDataGenerator

#import Cloudant
from cloudant.client import Cloudant
from cloudant.database import CloudantDatabase
from datetime import date

init_Base64 = 21
import io

diabetes_model = pickle.load(open("rf.sav","rb"))
cardio_model = pickle.load(open("model.sav", 'rb'))
my_model = load_model('model.h5')
df = pickle.load(open("dataframe.pkl", 'rb'))

status = 'dev'
dsn = "DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-dal09-11.services.dal.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=nsg21758;PWD=wjvgd9f-gfndl0cj;"

app = Flask(__name__)
app.secret_key = "grm project"
app.config["IMAGE_UPLOADS"] = "pred/images/"


def check_correct_login(cat):
    try:
        if(session["login"] == True and session["cat"] == cat):
            return True
        else:
            return False
    except:
        return False

@app.route("/",methods = ["GET","POST"])
@app.route("/login",methods = ["GET","POST"])
def login():
    try:
        if(session["login"]):
            pass
    except:
        session["login"] = False
        session["user_id"] = -1
        session["username"] = ""
        session["cat"] = ""

    if(request.method == "POST"):
        try:
            name = int(request.form["id"])
        except:
            flash("User ID is numeric !", category="error")
            return redirect(url_for('login'))
        password = request.form["password"]
        conn = ibm_db.connect(dsn,"","")
        sql = "select * from USERS where user_id = "+str(name)+" and password = '"+password+"';"
        res = ibm_db.exec_immediate(conn,sql)
        k = ibm_db.fetch_assoc(res)
        ibm_db.close(conn)
        if(k==False):
            flash("Invalid Credentials !!!",category="error")
            return render_template("login.html")

        else:
            session["user_id"] = k["USER_ID"]
            session["username"] = k["USERNAME"]
            session["cat"] = k["CATEGORY"]
            session["login"] = True
            cat = k["CATEGORY"]
            if(cat== "p"):
                session["cat"] = "p"
                return redirect(url_for('p_home'))
            elif(cat== "d"):
                session["cat"] = "d"
                return redirect(url_for('d_home'))
            elif(cat== "h"):
                session["cat"] = "h"
                return redirect(url_for('h_home'))

    return render_template("login.html")

@app.route("/logout",methods = ["GET","POST"])
def logout():
    session.pop("login",False)
    session.pop("cat","")
    session.pop("user_id",-1)
    flash("Successfully Logged Out!",category="success")
    return redirect(url_for('login'))


@app.route("/register",methods = ["GET","POST"])
def register():
    if(request.method == "POST"):
        conn = ibm_db.connect(dsn,"","")
        username = request.form["username"]
        sql = "select * from users where username ='"+username+"';"
        res = ibm_db.exec_immediate(conn,sql)
        k = ibm_db.fetch_assoc(res)
        if(k):
            flash("Username Already Exists")
            render_template("register.html")

        else:
            name = request.form["name"]
            age = int(request.form["age"])
            gender = request.form["gender"]
            height = int(request.form["height"])
            weight = int(request.form["weight"])
            password = request.form["password"]
            c_pass = request.form["confirm-password"]
            if(password == c_pass):
                sql = "insert into users(username,password,category) values('"+username+"','"+password+"','p');"
                res = ibm_db.exec_immediate(conn,sql)
                sql = "select * from users where username ='"+username+"';"
                res = ibm_db.exec_immediate(conn,sql)
                k = ibm_db.fetch_assoc(res)
                user_id = k["USER_ID"]
                sql = "insert into patient(user_id ,name,age,gender,height,weight) values ("+str(user_id)+",'"+str(name)+"',"+str(age)+",'"+str(gender)+"',"+str(height)+","+str(weight)+");"
                res = ibm_db.exec_immediate(conn,sql)
                flash("Registered Successfully! Please remember your User ID - "+str(user_id),category="success")
                return redirect(url_for("login"))

            else:
                flash("Passwords do not match !",category="error")
                return render_template("register.html")



    return render_template("register.html")

##############################################
##### PATIENT ################################
##############################################
@app.route("/patient")
def p_home():
    if(check_correct_login("p")):
        conn = ibm_db.connect(dsn,"","")
        sql = "select * from patient where user_id ="+str(session["user_id"])+";"
        res = ibm_db.exec_immediate(conn,sql)
        patient = ibm_db.fetch_assoc(res)
        sql = "select * from general_records where user_id ="+str(session["user_id"])+" order by date_of_check desc;"
        res = ibm_db.exec_immediate(conn,sql)
        recs = []
        k = ibm_db.fetch_assoc(res)
        
        while(k):
            recs.append(k)
            k = ibm_db.fetch_assoc(res)
        ibm_db.close(conn)
        return render_template("p_home.html",patient = patient,records = recs)

    else:
        flash("Patient Login Required !", category="error")
        return redirect(url_for('login'))
    
    
@app.route("/patient/news")
def news():
    if(check_correct_login("p")):
        result = requests.get("https://www.who.int/csr/don/archive/year/2020/en/")
        soup = BeautifulSoup(result.text,'html.parser')
        ul = soup.find('ul',class_ = 'auto_archive')
        dis = dict()
        li = ul.find_all('li')
        for i in li:
            name = i.span.text.split('â')[0].strip()
            link_date = [i.a["href"],i.a.text]
            #to order according to date(which is a string)
            if(dis.get(name)==None):
                dis[name] = link_date
            else:
                #append later dates to the end
                del(dis[name])
                dis[name] = link_date

        i = 0
        info_dis = dict()
        for d in dis.keys():
            url = "https://www.who.int"+dis[d][0]
            res = requests.get(url)
            soup = BeautifulSoup(res.text,'html.parser')
            p = soup.find_all('p')[1].span.text
            p += soup.find_all('p')[2].span.text
            dis[d].append(p)
            i+=1
            info_dis[d] = dis[d]
            if(i==5):
                break
            
            
        for key in info_dis.keys():
            del(dis[key])

        return render_template("p_news.html",news = dis,others = info_dis)
    
    else:
        flash("Patient Login Required !", category="error")
        return redirect(url_for('login'))


@app.route("/patient/details",methods = ["POST","GET"])
def patient_details():
    if(request.method == "POST"):
        allergies = request.form["allergies"]
        immunizations = request.form["immunizations"]
        observations = request.form["observations"]
        procedures = request.form["procedures"]
        careplans = request.form["careplans"]
        medications = request.form["medications"]
        current = request.form["current"]
        #flash("Patient Details Successfully Updated !!!",category="success")

        ## code to store this info in the database
	username= "e169e943-3ef0-47be-9cc8-7a6784a80114-bluemix"
        apikey= "HAltxJaD5Sr4VsbAVB4-rYLfWdrWOkW0fWuqzQFP_Ra_"
        client=Cloudant.iam(username,apikey,connect=True)
        
        db_name="patient_details"
        p_details=CloudantDatabase(client,db_name)
        
        date_v=str(date.today())
        encounter_id=str(session["user_id"])+'_'+date_v
        if p_details.exists():
            
            patient_document={
                    "allergies":allergies,
                    "immunizations":immunizations,
                    "observations":observations,
                    "procedures":procedures,
                    "careplans":careplans,
                    "medications":medications,
                    "current":current,
                    "encounter_id":encounter_id,
                    "date":date_v
                    }
            p_details.create_document(patient_document)
        #flash("Data Added Successfully !")
        client.disconnect()
        flash("Patient Details Successfully Updated !!!",category="success")

        return redirect(url_for("patient_details"))
        
    return render_template("p_details.html")


@app.route("/patient/value_based_prediction/",methods = ["POST","GET"])
def value_based_prediction():
    if(check_correct_login("p")):
        if(request.method == "POST"):
            age = int(request.form["age"])
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            ap_hi = float(request.form["ap_hi"])
            ap_lo = float(request.form["ap_lo"])
            chol = int(request.form["chol"])
            gluc = int(request.form["gluc"])
            gender = int(request.form["gender"])
            smoke = int(request.form["smoke"])
            alco = int(request.form["alco"])
            active = int(request.form["active"])
            enm = float(request.form["enm"])
            csd = float(request.form["csd"])
            dob = float(request.form["dob"])
            gd = float(request.form["gd"])
            dms = float(request.form["dms"])

            features = [enm,csd,dob,gd,dms,weight,ap_hi,ap_lo,smoke,gender]
            lab_dict=request.form
            dis_count= enm + csd + dob + gd + dms
            features.append(dis_count)
            fin_features=[np.array(features)]
            result=diabetes_model.predict(fin_features)
            result=result[0]
            diab = 0
            card = 0
            if result:
                diab = 1
            df["age"] = age
            if(gender == 0):
                df["gender"] = 1
            df["ap_hi"] = ap_hi
            df["ap_lo"] = ap_lo
            df["smoke"] = smoke
            df["alco"] = alco
            df["active"] = active
            bmi = weight/((height/100)**2)
            df["bmi"] = bmi
            df["pulse_pressure"] = ap_hi - ap_lo

            if(bmi<18.5):
                df["Underweight"] = 1
            elif(bmi<25):
                df["Healthy"] = 1
            elif(30<=bmi<35):
                df["Obese"] = 1
            elif(35<=bmi<40):
                df["Severly Obese"] = 1
            elif(35<=bmi<40):
                df["Abnormal"] = 1
            
            if(chol<200):
                df["chol_1"] = 1
            elif(chol>240):
                df["chol_3"] = 1

            if(gluc<115):
                df["gluc_1"] = 1
            elif(gluc>185):
                df["gluc_3"] = 1
            ypred = cardio_model.predict(df)
            if(ypred[0]>(0.5)):
                card = 1
            elif(ypred[0]>(0.3)):
                card = 2

            prob = int(ypred[0]*100)
            # return "<h1>"+str(card)+""+str(diab)+"</h1>"
            conn = ibm_db.connect(dsn,"","")
            from datetime import datetime
            now = datetime.now()

            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            lis=[session["user_id"],formatted_date,age,weight,height,ap_hi,ap_lo, chol, gluc, gender, smoke, alco,active, enm, csd, dob, gd, dms,card,diab]
            sql = "insert into history values %r;"%(tuple(lis),)
            res = ibm_db.exec_immediate(conn,sql)
            return render_template("result_vbp.html",card = card,diab = diab,prob=prob)


        return render_template("p_value_pred.html")

    else:
        flash("Patient Login Required !", category="error")
        return redirect(url_for('login'))




def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))
def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    desired_width, desired_height = 224, 224

    if width < desired_width:
   	 desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((100, 100))

    img = image.img_to_array(img)
    return img / 255.        

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/patient/image_based_prediction/",methods = ["POST","GET"])
def image_based_prediction():
    return render_template('form2.html')

@app.route('/predict',methods=['POST'])
def predict():
        image = request.files["image"]
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory("./pred", target_size=(224, 224), batch_size=5, class_mode='categorical', shuffle=False)        
        pred=my_model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
        os.remove(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))        
        

        """
        data = request.files['image']
        img = cv2.imread(data) 
        print(img)
        draw = img[init_Base64:]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        vect = vect.reshape(1, 1, 28, 28).astype('float32')
        """
        """
        npimg = np.fromfile(request.files['image'], np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        xnew = my_model.predict(img.reshape(1,224,224,1))
        """
        """#[0][0]
            #y_proba = my_model.predict(img)
            #y_classes = keras.np_utils.probas_to_classes(y_proba)
        y_prob = my_model.predict(img) 
        y_classes = y_prob.argmax(axis=-1)
        y_label = my_model.predict_classes(img, verbose=1)[0][0]
            #l.append(y_prob)
            #l.append(labels[1 if y_prob > 0.5 else 0])
            #predictedLabels[y_prob[0][0]] = labels[1 if y_prob > 0.5 else 0]
        predictedLabels[y_prob[0][0]] = labels[y_label]
            #print('prediction : ',xnew)
            #print('class : ',my_model.predict_classes(img, verbose=1))
            #print('y_classesmal : ',y_classes)
        #print('Classifying..')
    #c=request.form#.get("Weight")"""
        if pred[0][0]>=pred[0][1]:
            s="benign"
        else :
            s="malignant"
           
        
        return render_template('result2.html',result=s)   # show result in result.html

@app.route("/patient/history/",methods = ["POST","GET"])
def history():
    conn = ibm_db.connect(dsn,"","")
    """
    sql = "select * from history where PATIENT_ID=10003;"
    res = ibm_db.exec_immediate(conn,sql)
    #while ibm_db.fetch_row(res) != False:
        #print(ibm_db.result(res, 0))
    patient = ibm_db.fetch_assoc(res)
    """
    sql = "select * from history where PATIENT_ID ="+str(session["user_id"])+";"
    res = ibm_db.exec_immediate(conn,sql)
    recs = []
    k = ibm_db.fetch_assoc(res)
        
    while(k):
        recs.append(k)
        k = ibm_db.fetch_assoc(res)
    ibm_db.close(conn)    
    return render_template('history.html', records=recs)
@app.route("/patient/mental_health/",methods = ["POST","GET"])
def mental_health():
    if(check_correct_login("p")):
        if(request.method == "POST"):
            Intro1 = request.form["Intro1"]
            Intro2 = request.form["Intro2"]
            Intro3 = request.form["Intro3"]
            Intro4 = request.form["Intro5"]
            Intro5 = request.form["Intro4"]
            Intro6 = request.form["Intro6"]
            Intro7 = request.form["Intro7"]
            Intro8 = request.form["Intro8"]
            Intro9 = request.form["Intro9"]
            Intro10 = request.form["Intro10"]
            Intro11 = request.form["Intro11"]
            Intro12= request.form["Intro12"]
            Intro13 = request.form["Intro13"]
            Intro14 = request.form["Intro14"]
            prediction = nb.predict(vectorizer.transform([Intro1]))
            if prediction == 1:
                result="Positive"
            elif prediction == 0:
                result="Neutral"
            elif prediction == -1:
                result= "Negative"
            else:
                result="Nothing"
        
            

            return render_template('result_m.html', result=prediction)
    return render_template('mental_health.html')
##############################################
##### DOCTOR   ###############################
##############################################
@app.route("/doctor")
def d_home():
    if(check_correct_login("d")):
        return render_template("d_home.html")
    
    else:
        flash("Doctor Login Required !", category="error")
        return redirect(url_for('login'))
    
    
##############################################
##### HOSPITAL ###############################
##############################################
@app.route("/hospital")
def h_home():
    if(check_correct_login("h")):
        return render_template("h_home.html")

    else:
        flash("Hospital Staff Login Required !", category="error")
        return redirect(url_for('login'))
    
    
if __name__ == '__main__':
    if(status == "dev"):
        app.run(debug = True, use_reloader=False, threaded=False)
    else:
        app.run()

