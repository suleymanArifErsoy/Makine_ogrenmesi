import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Lojistik_regrasyon_Ornek/veri_kumesi.csv")

x = df.iloc[:,1:4].values # bağımsız değişkenler 
y = df.iloc[:,4:].values # bağımlı değişkenler (sınıf değişkenleri erkek ve kadın)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logR = LogisticRegression(random_state=0)
logR.fit(X_train,y_train)
y_pred = logR.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) 
print(cm)