

#* Çok fazla sayıdaki ham veriyi gruplara ayırmak için kullanılır 
#* Ham veriler hakkında daha önceden sizin sınıflandırma yapmamanız gerekmektedir 
#* Yapay zeka bu sınıflandırmayı bizim için yapar YANİ DENETİMSİZ ÖGRENME dediğimiz model grubuna girer YANİ X(Öznitelik) değerleri varken Y(Sınıflandırıcı) değerleri veri setimizde yoktur  
#* Unspervised learning modelinin en çok tercih edilenidir diyebiliriz 


#? Algoritmadaki K-Means'teki K değeri => kaç gruba ayrılacak olduğunu belirtmemiz gerekir .
#* Kaç tane grup varsa o kadar yuvarlak bir nokta oluşturulur ve uzayda bir birbirinden farklı konuma yerleştirilir 
#  yuvarlak noktalar tüm veriler ile uzaklığı oklid formülü ile hesaplandıktan sonra noktalara en yakın olan veriler o noktanın sınıflayıcısı olur 
#  Bu işlem bittikten sonra her bir yuvarlak noktaya ait veriler arasında ortalama ağırlık hesaplanır ve yuvarlak nokta hesaplanan agırlik noktasına dogru harelet eder
#  Tüm noktaların hareketi bittkten sonra tekrardan veriler üzerindeki uzaklıkları hesaplanır ve veriler hangi noktaya en yakın ise o gruba dahil edilir 
#  Bu işlemler taa ki noktaların ortalama agirlik hesaplaması yerine hareketi neredeyse hareket etmeyecek kadar olmasına kadar devam eder
#  Eğitim işlemi bittikten sonra tahminleme işlmeleri yapılılabilir.

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Proje6_Musteri_segmentasyonu/Avm_Musterileri.csv")
print(df.head())
df.rename(columns={"Annual Income (k$)":"Income"},inplace=True)
df.rename(columns={"Spending Score (1-100)":"Score"},inplace=True)

#* Verilerimizi Mutlaka nomalize etmek gerekiyor 0 ile 1 arasında normalize ediyor en kucuk veri => 0 , en buyuk veri => 1 olacak şekilde işlem yapılıyor 
scaler = MinMaxScaler()

scaler.fit(df[["Income"]])
df["Income"] = scaler.transform(df[["Income"]])
scaler.fit(df[["Score"]])
df["Score"] = scaler.transform(df[["Score"]])
print(df.head())

#* Modelimizi eğitelim 

k_menas_deger = range(1,10)

list_dist=[]
for k in k_menas_deger:
    kMeans_modeli = KMeans(n_clusters=k)
    kMeans_modeli.fit(df[["Income","Score"]])
    list_dist.append(kMeans_modeli.inertia_)
    



