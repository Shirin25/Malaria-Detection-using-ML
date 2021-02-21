import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib
import pickle

dataframe = pd.read_csv("csv/dataset4.csv")

x = dataframe.drop(["label"],axis=1)
y = dataframe["label"]
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)

joblib.dump(model,"rf_malaria_100_5")
pickle.dump(model,open("finalized_model.sav","wb"))

predictions = model.predict(x_test)

print(metrics.classification_report(predictions,y_test))