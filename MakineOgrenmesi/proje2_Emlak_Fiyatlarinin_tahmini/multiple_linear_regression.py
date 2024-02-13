import pandas as pd 
from sklearn import linear_model
import pickle 

dataFrame = pd.read_csv("proje2_Emlak_Fiyatlarinin_tahmini/multilinearregression.csv",sep=";")
print(dataFrame.head())

# modelimizi tanımlıyoruz 
reg = linear_model.LinearRegression()
reg.fit(dataFrame[["alan","odasayisi","binayasi"]].values,dataFrame["fiyat"].values)

# tahmini gerçekleştirelim 
tahmin = reg.predict([[230,4,10],[230,6,1],[355,3,20]])
tahmin1 = int(tahmin[0])
tahmin2 = int(tahmin[1])
tahmin3 = int(tahmin[2])
b1=reg.coef_[0] # katsayı öznitelik sayısı kadar vardır 
b2 = reg.coef_[1]
b3 = reg.coef_[2]
a = reg.intercept_ # ilk değer yani a 

x1 = 240 # alan 
x2 = 4   # oda sayisi  
x3 = 9   # bina yasi 
tahminDogrulama = int(a + b1 *x1 + b2*x2 + b3*x3)

print(f"1. {tahmin1}₺  -- 2. {tahmin2}TL -- 3. ${tahmin2}    4. Farkli yoldan {tahminDogrulama} ₺") 

#? Dosyayı pickle şeklinde kayıt edersek eğer , eğittimiz modeli sadece bu pickle dosyasını okuyarak tahmin yürütebliriz 

dosya_adi = "ev_fiyat_tahmin.pickle"
pickle.dump(reg,open(dosya_adi,"wb")) # Dosya içerisine verilerimizi yazdık 

#! pickle dosyası üzerinden eğitlmiş modelimiz üzerinden tahmin yürütme 
#! pickle_tahmin python dosyasında 


