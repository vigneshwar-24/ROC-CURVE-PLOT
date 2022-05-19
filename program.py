import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/default.csv"
data = pd.read_csv(url)


x=data[['student','balance','income']]

y=data['default']
x_train,x_test,y_train,y_test,= train_test_split(x,y,test_size=0.3,random_state=0)
log_regression= LogisticRegression()
log_regression.fit(x_train,y_train)
#define metrics
y_pred_proba=log_regression.predict_proba(x_test)[::,1]
fpr,tpr, _ = metrics.roc_curve(y_test,y_pred_proba)

plt.plot(fpr,tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

plt.show()

#define metrics
y_pred_proba=log_regression.predict_proba(x_test)[::,1]
fpr,tpr, _ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test,y_pred_proba)


plt.plot(fpr,tpr, label="AUC" + str(auc))
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc=4)

plt.show()
