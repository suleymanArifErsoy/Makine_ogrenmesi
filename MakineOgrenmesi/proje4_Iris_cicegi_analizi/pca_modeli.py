

#! Principal Component Analysis 

#? PCA ile çok boyutlu(feature) veri setlerini , veri setlerinin anlamını kaybetmeden daha az boyutlu bir hale getirmeyi amaçlıyoruz 

#? Boyut indirgemesini NEDEN yapıyoruz ???

# 1- Çok fazla boyutlu veri setlerini kolay görselleştirebilmek için 
#* insan gözüyle 2 veya 3 boyutlu verileri daha kolay irdeleyebilmek ve analizini yapmamızı sağlar 
# 2- Verileri sıkıştırmak için PCA uygulamasını yaparız 
#* Bazı veri setleri aşırı feature içerdiği zaman bunların depolanması ve analiz edilmesi zor olur. PCA uygulayarak bu işi kolaylauylaştırırız. 

#? PCA NASIL uygulanır 

#* eigenvektörler çıkarılır 
#* bu eigenvektörler daha sonra PCA 1.boyut ,PCA 2.boyut ... PCA N.boyut şeklinde N boyuta indirgenir. 
#* N = Düşürülmek istenen boyut sayısı 
#? Bu işlemler yapılırken varyans %90'dan fazla bir oranda korunmaya çalışılır. Yani minimum veri kaybı yaşanır. 

#! PCA NEREDE KULLANILIR
#* VERİ SIKIŞTIRMASI , FOTOĞRAF SIKIŞTIRMASI , YÜZ ALGILAMA (FACE RECOGNİTON) algoritmaları çalıştırılmadan önce bu işlem yapılarak insan yüzüne ait sadece en belirgin özellikler belirlenir

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

url = "proje4_Iris_cicegi_analizi/pca_iris.data"
df = pd.read_csv(url,names=['sepal length','sepal width','petal length','petal width','target'])

x = df.iloc[:,:-1]
y = df[["target"]]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#! Değerleri scale etmemiz gerekiyor. Çünkü her bir feature çok farklı boyutlarda   ve bunların yapay zeka tarafından eşit ağırlıklarda dengelenmesi gerekiyor:
#* Bu amaçla standart scale kullanılarak tüm verileri mean = 0 ve variance = 1 olacak şekilde değiştiriyoruz 
sc = StandardScaler()

X_scale = sc.fit_transform(x)

pca = PCA(n_components=2)
principalComponent = pca.fit_transform(X_scale)
principalDataFrame = pd.DataFrame(data=principalComponent,columns=["a","b"])

#* PCA uygulandıktan sonra son olarak PCA'li dataFrame içerisine sınıflandırmayı belirten target isimli column'u da ekliyoruz 
final_df = pd.concat([principalDataFrame,df[["target"]]],axis=1)


#? GRAFİK ÇİZİMİ 

cicek_turleri = ["Iris-setosa","Iris-virginica","Iris-versicolor"]
colors= ["red","green","blue"]
plt.xlabel("principal compenent 1")
plt.ylabel("principal component 2")

for cicek , col in zip(cicek_turleri,colors):
    dfTarget =final_df[df.target==cicek]
    plt.scatter(dfTarget[["a"]],dfTarget[["b"]],color=col)
    
plt.show()