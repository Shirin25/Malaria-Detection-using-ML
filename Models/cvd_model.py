#importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import jaccard_score,accuracy_score
import pickle

#read dataset
df = pd.read_csv("cardio_train.csv")

# check for null values
df.isnull().sum()
#no null values

#id has no correlation with cardio. drop the column
df = df.drop(columns = ['id'])

#convert age in days to years
def convert_days_to_years(days):
    return int(days//365.25)
df["age"] = df["age"].apply(convert_days_to_years)

#gender 2 - male, 1 - female. covert it to 0 - male and 1 - female for convinience
df.loc[df["gender"]==2,"gender"] = 0

# calculating bmi
df["bmi"] = df["weight"]/((df["height"]/100)**2)

#categorize bmi
def bmi_category(bmi):
    if(bmi<18.5):
        return "Underweight"
    if(bmi<25):
        return "Healthy"
    if(bmi<30):
        return "Overweight"
    if(bmi<35):
        return "Obese"
    if(bmi<40):
        return "Severly Obese"
    return "Abnormal"

df["bmi_cat"] = df["bmi"].apply(bmi_category)

#calculate pulse pressure
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

#get dummy columns for categorical variables
df_copy = pd.concat([df,pd.get_dummies(df["cholesterol"],prefix = "chol")],axis = 1)
df_copy.drop(columns = ["weight","height"],inplace=True)
df_copy.drop(columns = ["cholesterol"],inplace=True)
df_copy = pd.concat([df_copy,pd.get_dummies(df["bmi_cat"])],axis = 1)
df_copy.drop(columns = ["bmi_cat"],inplace=True)
df_copy = pd.concat([df_copy,pd.get_dummies(df["gluc"],prefix = "gluc")],axis = 1)
df_copy.drop(columns = ["gluc"],inplace=True)

#droping columns to avoid dummy variable trap
df_copy = df_copy.drop(columns = ['chol_2','Overweight','gluc_2'])
df_copy.head()



#removing outliers and abnormal data

#highest bp ever recorder was 370/350 and lowest was 50/20
df_copy = df_copy.loc[(df["ap_hi"]<=370)&(df["ap_hi"]>=50)&(df["ap_lo"]<=350)&(df["ap_hi"]>=20)]
#it is very rare to find someone with a bmi > 75 bmi < 12
df_copy = df_copy.loc[(df_copy["bmi"]<50)&(df_copy["bmi"]>12)]
#pulse pressure are usually between -20 and 80 (extreme cases)
df_copy = df_copy.loc[(df_copy["pulse_pressure"]<=80)& (df_copy["pulse_pressure"]>=-20)]
# removing cases when absolutely normal people suffer from cvd
rem = list(df_copy.loc[(df_copy["ap_hi"]>=110)&(df_copy["ap_hi"]<=130)&(df_copy["ap_lo"]>=70)&(df_copy["ap_lo"]<=90)&(df_copy["pulse_pressure"]<=50)&(df_copy["pulse_pressure"]>=30)&(df_copy["chol_1"]==1)&(df_copy["gluc_1"]==1)&(df_copy["bmi"]<22)&(df_copy["age"]<=50)&(df_copy["cardio"]==1)].index)
df_copy = df_copy.drop(index = rem)
# removing cases when absolutely abnormal people do not suffer from cvd
rem = list(df_copy.loc[(df_copy["ap_hi"]>=160)&(df_copy["ap_lo"]>=100)&(df_copy["pulse_pressure"]>=50)&(df_copy["chol_3"]==1)&(df_copy["gluc_3"]==1)&(df_copy["age"]>=50)&(df_copy["cardio"]==0)].index)
df_copy=df_copy.drop(index = rem)
#other outliers
rem = list(df_copy.loc[(df_copy["ap_hi"]<=125)&(df_copy["ap_lo"]<=85)&(df_copy["pulse_pressure"]<=40)&(df_copy["chol_1"]==1)&(df_copy["gluc_1"]==1)&(df_copy["active"]==1)&(df_copy["cardio"]==1)].index)
df_copy = df_copy.drop(index = rem)



df_new = df_copy.drop(columns = ['bmi'])
y = df_new['cardio']
x = df_new.drop(columns = ['cardio'])

#splitting into a training and testing set
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 3)

#building the classifier
lgb = LGBMClassifier()
lgb.fit(xtrain,ytrain)
pickle.dump(lgb,open('cvd_model.sav',"wb"))
pickle.dump(dataframe,open('dataframe.pkl',"wb"))
