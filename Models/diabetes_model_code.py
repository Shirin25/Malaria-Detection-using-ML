# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:46:46 2020

@author: acer
"""



# importing libraries

from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import  VotingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency
import pickle


# %% [code]
#select/not select categorical variable
#use chi-square test of independence

def test_independence(column,df):
        table=pd.crosstab(index=df["target"],columns=df[column])
        
        p_val=chi2_contingency(table)[1]
        print("p_value of target and {} is {}".format(column,p_val))
        return p_val

# %% [code]


# %% [code]

"""   DATA CLEANING AND PREROCESING   """

#mapping ICD-9 codes to diseases 
icd9code = {    
        '001-139': 'infectious and parasitic',
        '140-239': 'neoplasms',
        '240-279': 'endocrine, nutritional and metabolic, immunity disorders',
        '280-289': 'diseases of the blood and blood-forming organs',
        '290-319': 'mental disorders',
        '320-359': 'nervous system',
        '360-389': 'sense organs',
        '390-459': 'circulatory system',
        '460-519': 'respiratory system',
        '520-579': 'digestive system',
        '580-629': 'genitourinary system',
        '630-679': 'complications of pregnancy, childbirth, and the puerperium',
        '680-709': 'skin and subcutaneous tissue',
        '710-739': 'musculoskeletal system and connective tissue',
        '740-759': 'congenital anomalies',
        '760-779': 'certain conditions originating in the perinatal period',
        '780-799': 'symptoms, signs, and ill-defined conditions',
        '800-999': 'injury and poisoning',
        'E-V': 'external causes of injury and supplemental classification'
    }
print(type(icd9code))

#cleaning diagnosis dataset
data=pd.read_csv("ibm-diabetes/patient.csv",indec_col="PatientGuid")
df3=pd.read_csv("ibm-diabetes/diagnosis.csv.txt",index_col="PatientGuid")
df3_=df3[df3.columns[:-5]]
l=[]#.rename(icd9code)
c=0
for i in icd9code:
    l.append(icd9code[i])
    c+=1
l.append("Unknown")
#print(c)
df3_.columns=l
df3=pd.concat([df3_,df3[df3.columns[-5:]]],axis=1)
df3.columns
df3["target"]=data["target"]
#df3
#import seaborn as sns
#sns.heatmap(df3.corr())

#fig,ax=plt.subplots(18,1,figsize=(10,20),sharex=False)
c=0
df3_=df3[df3.columns[:-6]]#[df3[df3.columns[:-6]]>0]=1
df3_[df3_>0]=1
df3_["target"]=df3["target"]

for i in df3_.columns[:-1]:
    plt.figure()
    print(i)
    plt.title(i)
    sns.countplot(x=df3_['target'],hue=df3_[i])
    c+=1

df3_

# %% [code]
#selecting most influential attributes from diagnosis 
sel=[]
for i in data.columns[:10]:
    k=test_independence(i,data)
    if k<0.05:
          sel.append(i)

# %% [code]
# Collating,Cleaning and pre-processing different datasets


data_dis=data[sel]
data_dis.drop(['symptoms, signs, and ill-defined conditions','external causes of injury and supplemental classification','complications of pregnancy, childbirth, and the puerperium'],axis=1,inplace=True)
data_dis

len(data["target"][data["target"]==1])



df3.columns

#df3["AcuteCount"]

#df3_["age"]=patient["Age"]
sns.boxplot(y=df3_["age"],x=df3_['target'])


#

df1=pd.read_csv("d1.csv",index_col="PatientGuid")

df1=df1.loc[~df1.index.duplicated('last')]
df1.drop(["State","PracticeGuid","Division","SmokeEffectiveYear","YearOfBirth","SmokingStatus_Description","SmokingStatus_NISTCode"],inplace=True,axis=1)
df1.fillna(0,inplace=True)
d=df1.loc[df1["Smoker"]!=True]
d=d.loc[d["Smoker"]!=False]

enc=LabelEncoder()
df1["Gender"]=enc.fit_transform(df1["Gender"]) # 0 : Female , 1: Male
df1["Smoker"]=enc.fit_transform(df1["Smoker"]) # 0,Na : False , 1 : True
sns.distplot(df1["age"])

transcript=pd.read_csv("ibm-diabetes/transcript.csv",index_col="Patient_Guid")
data=pd.concat([df3_,transcript],axis=1)
#data.drop(['Temperature_Mean','Temperature_Change','Temperature_Min','Temperature_Std','target','Height_Max','Weight_Max',])
for i in data.columns:
    if re.match("[\w]*Max",i):
        #print(i)
        data.drop(i,inplace=True,axis=1)
    if re.match("[\w]*Min",i):
        #print(i)
        data.drop(i,inplace=True,axis=1)
    if re.match("[\w]*Std",i):
        #print(i)
        data.drop(i,inplace=True,axis=1)    

data.columns

data=pd.read_csv("data.csv",index_col="PatientGuid")
data

data.columns

#DROPPING IRRELEVANT COLUMNS

pd.crosstab(index=data["target"],columns=data['complications of pregnancy, childbirth, and the puerperium'])

y=data["target"]
data.drop("target",axis=1,inplace=True)
data.drop(['complications of pregnancy, childbirth, and the puerperium','certain conditions originating in the perinatal period'],axis=1,inplace=True)
data.drop(['Unknown','injury and poisoning','symptoms, signs, and ill-defined conditions'])

data.columns
data.to_csv("ibm-diabetes.csv")

# %% [code]
data=pd.read_csv("ibm-diabetes.csv")

# %% [code]
#final selected diseases
disease_selected=['endocrine, nutritional and metabolic, immunity disorders','circulatory system','diseases of the blood and blood-forming organs','genitourinary system','musculoskeletal system and connective tissue']
sel_dis=df3_[disease_selected]
#make a new dataframe containg only selected diseases
sel_dis["Disease_count"]=sel_dis.sum(axis=1)
sel_dis['tr']=df3_["target"]
#sns.countplot(sel_dis['Disease_count'],hue=sel_dis['tr'])

# take only numerical columns
numerical_d=data[['Weight_Mean','SystolicBP_Mean','DiastolicBP_Mean','Smoker','Gender','AcuteCount']]
len([[numerical_d["AcuteCount"]==sel_dis["Disease_count"]]==True])
#concatenate the columns
d=pd.concat([sel_dis,numerical_d],axis=1)
d.drop('tr',axis=1,inplace=True)
d["target"]=df3_["target"]
d.columns
#save selected dataframe as csv file

d.to_csv("final_sel.csv")

#read newly created final_sel.csv file
data=pd.read_csv("final_sel.csv",index_col="PatientGuid")

data.columns

sns.boxplot(y=data["Height_Mean"],x=data["target"])


# converting height from numeric to categorical variable
data.loc[(data["target"]==1),"Height_Mean"].describe()
h_1=data.loc[data["target"][data["target"]==1].index]["Height_Mean"]
h_1[h_1>67]=65.864205
data.loc[data["target"][data["target"]==1].index]["Height_Mean"]=h_1
data.loc[data["target"][data["target"]==1].index]["Height_Mean"].describe()


data.loc[ (data["target"]==0) & (data["Weight_Mean"]<31),"Weight_Mean"]= 52.011883

data.loc[ (data["target"]==1) ,"Weight_Mean"].describe()

[data["Height_Mean"]>67]# or data["Height_Mean"]>70]


#reading from newly created csv file
data=pd.read_csv("/kaggle/input/ibm-diabetes/final_sel.csv",index_col='PatientGuid')
data.columns


#dc=data["Disease_count"]
y=data["target"]
data.drop(["AcuteCount","Disease_count","target"],axis=1,inplace=True)
data["Disease_count"]=dc
#data.insert(loc=)

X=data

# train test split
from sklearn.model_selection import train_test_split
dis_tr, dis_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)


# %% [code]
#outlier-detection using PCA

pca=PCA(n_components=2)
trans=pd.DataFrame(pca.fit_transform(X))
trans.index=X.index
pos_idx=y[y==1].index
plt.plot(trans[0],trans[1])
plt.plot(trans.loc[pos_idx,0],trans.loc[pos_idx,1])


### Modelling Cleaned data



# %% [code]
#baseline Support Vector classifier

svc=SVC()
param_grid={'kernel':('linear','rbf'),'class_weight':['balanced']}
clf=GridSearchCV(svc,param_grid,cv=5,scoring='recall',return_train_score=True)#{'recall':recall_score,'prec':precision_score})
clf.fit(dis_tr,y_tr)
clf.best_estimator_
clf.best_score_
clf

# %% [code]
#Linear ensemble
vc=VotingClassifier([("lr",lr),("svc",svc)],voting='hard')
vc.fit(dis_tr,y_tr)
y_pr=vc.predict(dis_ts)
classification_report(y_ts,y_pr,output_dict=True)

# %% [code]
lr=LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear',penalty='l1',fit_intercept=False)
lr.fit(dis_tr,y_tr)
y_pr=lr.predict(dis_ts)
classification_report(y_ts,y_pr,output_dict=True)

# %% [code]
#logistic roc-auc curve
fpr,tpr,tr=roc_curve(y_ts.values,lr.predict_proba(dis_ts)[:,1])
plt.plot(fpr,tpr)

# %% [code]
"""
from sklearn.svm import SVC
from sklearn.metrics import classification_report
svc=SVC(kernel='linear',degree=10,probability=True,class_weight='balanced',gamma='scale')#,regularization='l1')
#cols=sel_columns_model(rf)
svc.fit(dis_tr,y_tr)
#cv_results=cross_validate(svc,dis_tr,y_tr,cv=5,scoring=('recall','f1'))
#cv_results
y_pr=svc.predict(dis_ts)
classification_report(y_ts,y_pr,output_dict=True)
#sel_columns_model(svc)
"""

# %% [code]
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
tr=DecisionTreeClassifier(criterion='gini',max_depth=5,class_weight='balanced')
abc=AdaBoostClassifier(n_estimators=200,learning_rate=5,random_state=23)
abc.fit(dis_tr,y_tr)
y_pr=abc.predict(dis_ts)
classification_report(y_ts,y_pr,output_dict=True)
#sel_columns_model(abc)
"""

# %% [code]
#fitting randomforestclassifier
rf=RandomForestClassifier(n_estimators=100,max_features='auto',max_depth=2,class_weight='balanced',criterion='entropy',random_state=123)
rf.fit(dis_tr,y_tr)
y_pr=rf.predict(dis_ts)
classification_report(y_ts,y_pr,output_dict=True)
#cv_results=cross_validate(rf,dis_tr,y_tr,cv=5,scoring=('recall','f1'))
#cv_results
#assessing performance of randomforest


#pickling randomforest model
#!mkdir temp
path='./temp/rf.sav'
pickle.dump(rf,open(path,"wb"))

# %% [code]
#random forest roc-auc curve
fpr,tpr,tr=roc_curve(y_ts.values,lr.predict_proba(dis_ts)[:,1])
plt.plot(fpr,tpr)

# %% [code]
#opening pickled model

with open('rf-model/rf (1).sav','rb') as f:
    model=pickle.load(f)

#loading full dataset again

data=pd.read_csv("final_sel.csv",index="PatientGuid")
y=data["target"]
X=data.drop("target",axis=1)
y_pr_full=model.predict(X)

#final random forest model results:
# recall:0.83456721
# f-score:0.55789243
