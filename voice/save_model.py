from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
#from first import getdata
import pickle


voice_data = pd.read_csv(r'C:\Users\ASHAROX\Downloads\voice.csv')

#encoding the label
voice_data['label'].replace(to_replace={'male':1,'female':0},inplace = True)
#seperating the dataset to independent and dependent variables
y = voice_data.label.values.reshape(-1,1)
x = voice_data.drop(["label"], axis=1)

#transforming the train and test data and training the dataset
for i in range (0,1000):
  x_train,x_test,y_train,y_test = \
  train_test_split(x,y,test_size=0.3,random_state = i)
  stdc = StandardScaler()
  x_train_std = stdc.fit_transform(x_train)
  x_test_std = stdc.fit_transform(x_test)
  logit = LogisticRegression(C=1)
  logit.fit(x_train_std,y_train.ravel())
  y_pred = logit.predict(x_train_std)
  #print(y_pred)
  score = logit.score(x_train_std,y_train)
  #print(score)

#saving the model using picklepip install -U scikit-learn
#Saving the Model
pickle_out = open("logit.pkl", "wb") 
pickle.dump(logit, pickle_out) 
pickle_out.close()