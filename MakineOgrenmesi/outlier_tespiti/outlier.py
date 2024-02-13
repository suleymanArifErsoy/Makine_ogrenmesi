import pandas as pd 

#! Oulier nedir : Bir veri seti içerisindeki sapan veya aykırı bir değerdir 
#? örnek olarak emlakcılık üzerinden bir makine ogrenmesi projesi geliştireceğiz diyelim .
#? 150 m2 yazacagımız yere bir fazla sıfır ekleyerek 1500 m2 yazdığımızı varsayalım
#? makine öğrenmesi algoritması bu hatadan dolayı yanlış bir hesaplama yapar ve kullanıcıyı yanıltır . 
#? bu tür insan kaynaklı hataların genel ismi oulier olarak tanımlanır .

# Outlier'ın nedenleri ;
# 1- İnsan kaynaklı hatalar (hatalı veri seti girişleri)
# 2- Cihazlardan kaynaklanan hatalar (Örneğin ölçüm cihazların nadiren de olsa yanlış ölçüm yapması)

#! Outlier nasıl hesaplanır 

# Q1 (Percentile %25) => Veri setindeki sayıları büyükten küçüğe doğru sıraladığımızda , en küçük sayı ile ortanca değer arasındaki ortanca sayıdır
# Q3 (Percentile %75) => Veri setindeki sayıları büyükten küçüğe doğru sıraladığımızda ; ortanca sayı ile en büyük değer arasındaki ortanca sayıdır.
# IQR = Q3 - Q1
# Alt_limit = Q1 - 1.5 * IQR
# Üst_limit = Q3 + 1.5 * IQR

df = pd.read_csv("outlier_tespiti/outlier_ornek_veriseti.csv",sep=";")

print(df.describe()) #? dataFrame ile ilgili bazı istatistik verilerini yazar (medyan , ortanca ,standart sapma,min , max ,count)

#? Q1 (Percentile hesaplama)
Q1 = df["boy"].quantile(0.25)

#? Q3 (Percentile hesaplama)
Q3 = df.boy.quantile(0.75)

#? IQR değeri hesaplama
IQR_degeri = Q3 - Q1

#? Alt limit ve ust limit hesaplama
alt_limit = Q1 - 1.5 * IQR_degeri
ust_limit = Q3 + 1.5 * IQR_degeri

#? alt ve ust değerler dışında kalan veriler 
outlier_data = df[(df["boy"] < alt_limit) | (df.boy > ust_limit)]

#? filtreleme işlemi  => Alt_limit ile Üst_limit arasındaki değerleri aldık 

#df_filtreleme = df[ (df.boy >alt_limit) & (df.boy < ust_limit)]
df_filtreleme = df.loc[(df["boy"]>alt_limit) & (df["boy"]<ust_limit)]
print(df_filtreleme)

