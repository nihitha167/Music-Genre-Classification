import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# for 30 sec dataset
music_30sec = pd.read_csv('features_30_sec.csv')
encoder = preprocessing.LabelEncoder()
music_30sec['label'] = encoder.fit_transform(music_30sec['label'])
X_30s = music_30sec.drop(['label','filename'],axis=1)
y_30s = music_30sec['label']
cols = X_30s.columns
minmax = preprocessing.MinMaxScaler()
np_scaled = minmax.fit_transform(X_30s)
X_30sec = pd.DataFrame(np_scaled, columns = cols)
X_train30, X_test30, y_train30, y_test30 = train_test_split(X_30sec, y_30s,test_size=0.3,random_state=111)
X_train30.shape, X_test30.shape, y_train30.shape, y_test30.shape

ranforcl = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model = ranforcl.fit(X_train30, y_train30)
preds_train = ranforcl.predict(X_train30)
preds = ranforcl.predict(X_test30)
print("train accuracy",accuracy_score(y_train30, preds_train))
print("test accuracy",accuracy_score(y_test30, preds))

figure = plt.figure(figsize=(5, 5))
cm = confusion_matrix(y_test30,preds,labels = [0,1,2,3,4,5,6,7,8,9])
cm_normalized = np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
sns.heatmap(cm_normalized,cmap=plt.cm.Reds,annot= True,xticklabels = [0,1,2,3,4,5,6,7,8,9],yticklabels = [0,1,2,3,4,5,6,7,8,9])

#plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix for 30 seconds data, Random forest model')
plt.show()

print(classification_report(y_test30,preds))

# for 3 sec dataset
music_3sec = pd.read_csv('features_3_sec.csv')
encoder = preprocessing.LabelEncoder()
music_3sec['label'] = encoder.fit_transform(music_3sec['label'])
X_3s = music_3sec.drop(['label','filename'],axis=1)
y_3s = music_3sec['label']
cols = X_3s.columns
minmax = preprocessing.MinMaxScaler()
np_scaled = minmax.fit_transform(X_3s)
X_3sec = pd.DataFrame(np_scaled, columns = cols)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_3sec, y_3s,test_size=0.3,random_state=111)
X_train3.shape, X_test3.shape, y_train3.shape, y_test3.shape

ranforcl = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model = ranforcl.fit(X_train3, y_train3)
preds_train = ranforcl.predict(X_train3)
preds = ranforcl.predict(X_test3)
print("train accuracy",accuracy_score(y_train3, preds_train))
print("test accuracy",accuracy_score(y_test3, preds))



figure = plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test3,preds,labels = [0,1,2,3,4,5,6,7,8,9])
cm_normalized = np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
sns.heatmap(cm_normalized,cmap=plt.cm.Reds,annot= True,xticklabels = [0,1,2,3,4,5,6,7,8,9],yticklabels = [0,1,2,3,4,5,6,7,8,9])

#plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix for 3 seconds data, Random forest Model')
plt.show()

print(classification_report(y_test3,preds))