import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#! K-NN Algoritması  K - Nearest Neighbours  K - En yakın komşular

#? Daha önceden elde edilen noktalara bakarak yeni bir nokta geldiği zaman bu noktanın hangi noktaya ait olduğunu sınıflandırmak için kullanılan bir algoritmadır 
# Outcome : 1 -> Şeker hastaesı 
# Outcome : 0 -> Sağlıklı

df = pd.read_csv("Proje1_seker_hastaliği_tahmini/diabetes.csv") 

seker_hastalari = df[df.Outcome == 1]
saglikli_insanlar = df[df.Outcome == 0]

"""
plt.figure(figsize=(14,7))
plt.scatter(saglikli_insanlar.Age,saglikli_insanlar["Glucose"],color="green",label = "saglikli",alpha=0.4)
plt.scatter(seker_hastalari.Age,seker_hastalari.Glucose,color = "red", label = "Seker Hastasi",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")

plt.legend()
plt.show()
    
"""



# x ve y eksenlerini belirleyelim 
y = df.Outcome.values
x_ham_veri = df.drop(["Outcome"],axis=1)

#? Outcome sütununu bırakıp sadece bağımsız değerleri alıyoruz 
#! Çünkü KNN algoritması sadece x değerleri için bir hesaplama yapar 

#? normalizasyon yapıyoruz x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz 
#? Eğer bu şekilde normalizasyon yapmazsak yüksek rakamlar küçük rakamları ezer ve KNN algoritması yanılabilir

x = (x_ham_veri - np.min(x_ham_veri) ) / (np.max(x_ham_veri) - np.min(x_ham_veri)) # normalizasyon işlemi


# randomState => veri setindeki değerleri eğitim ve test işlemine sokmak için random olarak seçer
# test_size => Veri setinin %20'sini test için kalanı (%80)'nı eğitim için kullan 
x_train , x_test , y_train , y_test =train_test_split(x , y , test_size=0.20 ,random_state=1) 

# Veri setimizin eğitim ve test işlemi olamk üzere ikiye random bir şekilde ayırmış olduk

# knn modelimizi oluşturuyoruz 

knn = KNeighborsClassifier(n_neighbors= 28) # n_neighbours = k 
knn.fit(x_train,y_train) # fit =>  Modeli eğit
tahmin = knn.predict(x_test) # tahmin 



enBuyuk = 0
# Optimum K değeri nasıl belirliyecegiz 

for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors= k)
    knn_yeni.fit(x_train,y_train)
    
    
    score = knn.score(x_test,y_test)*100
    if score > enBuyuk:
            enBuyuk=score
    print(f"{k} - Doğruluk oranlari : % {knn_yeni.score(x_test,y_test)*100}")
print(f"En yuksek Dogruluk orani {enBuyuk}  {tahmin}")