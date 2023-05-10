# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:15:57 2023

@author: bisos
"""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
df = pd.read_csv("../MLRegModel/car_data.csv")

sns.heatmap(df.corr(),annot=True);

#VERİ ÖN İŞLEM
sns.set()
cols = ['Selling_Price', 'Year', 'Present_Price', 'Driven_kms', 'Fuel_Type','Owner','Transmission','Selling_type']
sns.pairplot(df[cols], height = 3.5)
plt.show();
#df =df.iloc[:,1:len(df)]
print(df.head())
print(df.shape)
print(df.info())


#ekisik veri varsa kontrol eder
df.isnull().sum()


# kategorik verilerin dağılımının kontrol edilmesi
df.Fuel_Type.value_counts()
df.Selling_type.value_counts()
df.Transmission.value_counts()


#Kategorik Verilerin Kodlanması
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Selling_type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

# data frame kategoriik değişkenler düneledikten sonra kontrol etmek
df.head()

""""""




#Verileri ve Hedefi bölme

#iki hedef değişkenlerimiz arasında ilişki
sns.jointplot(x="Selling_Price", y="Present_Price", data=df, kind="reg")

#bağımsız değişken veri site
X = df.drop(['Car_Name','Selling_Price'],axis=1) #bağımsız değişkenler

print(X)
Y = df['Selling_Price']  #bağımLİ değişken
print(Y)

#Eğitime ve Test verilerini ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

""""""




#Linear Regression STRAT...
lin_reg= LinearRegression()
model = lin_reg.fit(X_train,Y_train)
model

#Model Evaluation

#tahmin train site "prediction "
training_data_prediction = lin_reg.predict(X_train)
training_data_prediction
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
error_score


#gravik ile gereçek ve tahmin fiyatlar
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

#tahmin test datasite  "prediction "
test_data_prediction = lin_reg.predict(X_test)
test_data_prediction

#tahmin formul elmized oldu model =model.coef * değişken + model.intercept.
model.coef_ #x değişkenleri b1 ,b2,…
model.intercept_ #sabit b0

# R squared Error  tamin hatası
error_score = metrics.r2_score(Y_test, test_data_prediction)
error_score


#test verisite grafikleri çezereiz
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

#TAHMİN ÖRENİĞİ
"""Bu kod parçası, lineer regresyon modelinin kullanımını örneklemektedir. Belirli bir girdiye
dayanarak araba satış fiyatını tahmin etmek için oluşturulan modeli kullanarak bir tahmin yapar
. Tahmin, 2004 model, 6.55 present price, 700 km'de, dizel yakıt tipi, manuel şanzıman, tek sahip
 ve bayi tarafından satılmış bir aracın satış fiyatıdır. Tahmin sonucu 1.74387848 değerindedir."""


Fiyat_prediction = lin_reg.predict([[2020, 6.55 ,700 ,1,1,1,1 ]])
Fiyat_prediction

""""""





# loading the Lasso Regression model START...
lass_reg_model = Lasso()

lass_reg_model.fit(X_train,Y_train)


#Model Evaluation

# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

#Visualize the actual prices and Predicted prices

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# prediction on Training data
Lass_test_data_prediction = lass_reg_model.predict(X_test)


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

#Visualize the actual prices and Predicted prices

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

lass_reg_model.coef_   #değişken katsayıları
lass_reg_model.intercept_  #sabit sayı



from sklearn.model_selection import cross_val_score
cross_val_score(model , X_train, Y_train, cv = 10, scoring="neg_mean_squared_error")

Lass_Fiyat_prediction = lass_reg_model.predict([[2004, 6.55 ,700 ,1,1,1,1 ]])
Lass_Fiyat_prediction




