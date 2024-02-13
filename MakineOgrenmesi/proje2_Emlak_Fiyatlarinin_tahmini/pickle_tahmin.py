import pickle 

#! Modelimizi eğitip bir pickle dosyası üzerine kaydedersek sadece yazılmış olan pickle dosyasını açıp modelimiz üzerinden tahmin işlemi yapabiliriz. 
modelimiz=pickle.load(open("ev_fiyat_tahmin.pickle","rb")) # Okuma modunda pickle dosyamızı açtık 
tahmin=modelimiz.predict([[230,3,10]])
print(tahmin) 

