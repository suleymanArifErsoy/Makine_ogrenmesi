import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 


# y = a + b1*x1^2 + b2*x2^2 ... bN*xN^N
#! polinom regresyon veya farkı bir regresyon çeşitlerinden biri seçmek için öncellikle veri setimizdeki dağılıma göre bir seçim yapacağız.

# veri setlerinde çalışanlar için bir maaş skalası vardır 
#? yeni bir yönetici tanımlanırsa ve bu yönetici 4 ve 5 deneyim seviyesi arasında 4.5 senelik iş deneyimi olan bir departman açıldığında 
# 4.5 deneyim süresine sahip eleman ne kadar maas alması gerektiğini hesaplayacagız 


df = pd.read_csv("Proje3_HR_departmani_maas_hesaplama/polynomial.csv", sep=";")
print(df.head())



x = df[["deneyim"]].values
y = df["maas"].values



#? matplotlib ile veri seti gorselleştirildiğinde , veriler doğrusal bir şekilde dağılmıyor ve polinominal bir dağılım olduğu görülüyor 
#? eğer biz burada linear regresyon uygularsak verimli bir sonuç alamayız 



'''
plt.figure(figsize=(12,5))
plt.title("HR departman maas Gostergesi")
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlabel("deneyim")
plt.ylabel("Maas bilgisi")
plt.scatter(x,y,label="maas")

#! Linear regresion ile tahmin 
linear_reg = LinearRegression()
linear_reg.fit(x , y)
x_ekseni = df["deneyim"]
y_eksen_tahmin = linear_reg.predict(df[["deneyim"]].values)
plt.plot(x_ekseni,y_eksen_tahmin,color="red",label ="linear_regression")
plt.savefig("Linear_regression_gosterimi.png",dpi=50)
plt.show()
'''
polinom_regresyon = PolynomialFeatures(degree = 4)

x_polinom_reg=polinom_regresyon.fit_transform(df[["deneyim"]])


linear_reg = LinearRegression()

#? polinom regression modelinde x değerlerini fit_transform
linear_reg.fit(x_polinom_reg , y) 

y_haed = linear_reg.predict(x_polinom_reg)
plt.scatter(x,y,color="green")
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.plot(x,y_haed,color="red",label="polinom_reg")
plt.legend()
plt.savefig("polinom_reg.png",dpi=350)
plt.show()

# 4.5 iş deneyimine ait maas tahmini hesaplaması 

x_polinom_reg1 = polinom_regresyon.fit_transform([[4.5]])
print(linear_reg.predict(x_polinom_reg1))

