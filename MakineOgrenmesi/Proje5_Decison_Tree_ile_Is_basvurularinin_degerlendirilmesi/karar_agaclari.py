import pandas as pd
from sklearn import tree

df = pd.read_csv("Proje5_Decison_Tree_ile_Is_basvurularinin_degerlendirilmesi/DecisionTreesClassificationDataSet.csv")

#* for dongusuyle işlemi daha hızlı ve kolay yapmak için veri setindeki Y ve N değerlerini alan sutunları yanyana almak için iki sutunu yer değiştirdik
yeni_sutun_isimleri = {df.columns[1]:"Egitim Seviyesi", df.columns[3]:"SuanCalisiyor?"}
duzeltme_mapping= {"Y":1,"N":0}
duzeltme_mapping_egitim ={"BS":0,"MS":1,"PhD":2} # BS = lisans , MS = master , PhD = doktora 

#* for dongusu ile daha kolay bir şekilde yapmak için sadece 0 ve 1 den oluşan verileri aktarmak için columların yerlerini değiştirme
#? Skelearn algoritması ile egitilecek olan verilerin değerlerini sayısal bir değer olması şart bunun için tüm satırlardaki verileri anlamını kaybetmeyecek bir şekilde sayısallaştırıyoruz

df["Egitim Seviyesi"] , df["SuanCalisiyor?"] = df["SuanCalisiyor?"],df["Egitim Seviyesi"]# sutunları yer değiştirildi.
df.rename(columns=yeni_sutun_isimleri,inplace=True)# sutunlarin değerleri yer değiştirildi ama sutun isimleri değişmedi bunun için sutunları yeniden isimlendiriyoruz

sutun_isimleri=df.columns
# df içindeki bilgileri sklearn icerisinde bulunan tree.DecisionTreeClassfier() fonksiyonundaki fit etme işleminde istenen şekilde düzeltme işlemi  
for i in range(4):
    df[sutun_isimleri[i+3]] =df[sutun_isimleri[i+3]].map(duzeltme_mapping) #* replace() metodu da kullanılabilir ama map() fonk değer atanmayan özelliklere NaN değerini atar
df["Egitim Seviyesi"] = df["Egitim Seviyesi"].map(duzeltme_mapping_egitim)     
print(df)

# Eğitim için X(Öznitelik değerleri) ve Y(Sınıflandırıcı) değerlerini alalım 
Y = df["IseAlindi"].values # values ile veri setindeki değerleri dizi şeklinde saklıyoruz
X = df.drop(["IseAlindi"],axis=1).values # X değerlerini oluşturan veriler Y dışındaki tüm verilerdir. bu nedenle Y (sınıflandırıcı) sutununu silip tum dataFrame'i X değişkenine aktardık


#* Decison Tree'mizi oluşturuyoruz 

clf = tree.DecisionTreeClassifier() # Karar Ağacımızı oluşturduk 
clf = clf.fit(X,Y) # Karar Agacında fit() metodu ile ilk önce öznitelik verileri daha sonrasında Y sınıflandırıcı verilerini modele egitmek için gönderdik
sonuc = clf.predict([[0,2,0,0,1,1]]) # modelimiz eğitildi . Bundan sonra gerekli parametreleri yazarak tahmin değerlerini yapabiliriz
if sonuc == 1:
    print("ise alindi")
else:
    print("ise alinmadi")     
# decison tree Egitilmiş modeline gore ise alındı 

