# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:14:26 2023

@author: bisos
"""

# telfon fiyatlar tehmin
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge , ElasticNet , ElasticNetCV, RidgeCV,LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV



#VERİ ÖN İŞLEM
dfPhon= pd.read_csv("../MLRegModel/TelData.csv")
dfPhon.head()
dfPhon.info()
dfPhon.isnull().sum()
dfPhon.describe()

""""""


#VeriGöresleştirme
sns.pairplot(dfPhon,hue='price_range')
sns.pointplot(y="int_memory", x="price_range", data=dfPhon)

#% of Phones which support 4G
labels4g = ["4G-supported",'Not supported']
values4g = dfPhon['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

""""""



#veriler tese ve tearn ayırma

XPhon=dfPhon.drop('price_range',axis=1)
yPhon=dfPhon['price_range']

XPhon.info()
X_train_P, X_test_P, Y_train_P, Y_test_P = train_test_split(XPhon, yPhon, test_size=0.33, random_state=101)

""""""



#LogisticRegression STRAT....
logmodel = LogisticRegression()

logmodel.fit(X_train_P,Y_train_P)


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)


logmodel.score(X_test_P,Y_test_P)
logmodel.cofe_
LogisticRegression
#model fitting
enet_model = ElasticNet().fit(X_train_P, Y_train_P)
enet_model.coef_
enet_model.intercept_

#tahmin

enet_model.predict(X_train_P)[:10]

#test setinin Tahmini bulmak için
enet_model.predict(X_test_P)[:10]

y_pred_p = enet_model.predict(X_test_P)

#test haat hesaplama RMS

np.sqrt(mean_squared_error(Y_test_P, y_pred_p))
r2_score(Y_test_P, y_pred_p)

""""""


#ElasticNetCV START...
#optimum lambda değeri  ElasticNetCV Method ile buluruz
enet_cv_model = ElasticNetCV(cv = 10 ).fit(X_train_P, Y_train_P)
enet_cv_model.alpha_

enet_cv_model.intercept_
enet_cv_model.coef_
#ElasticNet Regresyon Model Tuning

#final model
enet_tuned =ElasticNet(alpha= enet_cv_model.alpha_).fit(X_train_P, Y_train_P)
enet_tuned

y_pred_enet_mode_p = enet_tuned.predict(X_test_P)
#VERİBİLİMCİ OLARAK EN düşük  HATA VEREN MODEL KULLANMAMIZ GEREKEİR
np.sqrt(mean_squared_error(Y_test_P,y_pred_enet_mode_p))






#Ridge Model Start....
ridge_model = Ridge(alpha =0.5).fit( X_train_P, Y_train_P)
ridge_model
ridge_model.coef_

ridge_model.intercept_


 #optimum alfa parametresini bulmak için rastgele bir alfa değerleri kümesi oluşturuyoruz
np.linspace(10,-2,100)
lambadar=  10** np.linspace(10,-2,100)*0.05



#Tahmin
ridge_model = Ridge().fit(X_train_P, Y_train_P)
#herhangi bir alfa parametresi belirtmeden train veri seti için tahmin değerlerini y_pred olarak kaydediyoruz.
y_pred_ridge_p = ridge_model.predict(X_test_P)

#Bu koşullar altında tahmin edilen verilere göre, Kök Ortalama Kare Hata değerini aşağıdaki gibi hesaplıyoruz.
RMSE_red = np.sqrt(mean_squared_error(Y_train_P, y_pred_ridge_p))
RMSE_red

#Model Tunnig Start...
#Alfha site oluşturma :
lambda_values1 = np.random.randint(0,1000,100)
lambda_values_2 = 10**np.linspace(10,-2,100)*0.5

#nasıl uygun lambda değer seçeriz
#Ridge Regression ile ayarlanmış modeli oluşturmak için RidgeCV kullanıyoruz  çeiştli lambda değeri hesablaerken çeşitli deneme yapıp ve çeşitlli hata değerleri hesapladık

#farklı lambda değeri kullanarak farklı alpha değeri elde bilirizi
ridgecv = RidgeCV(alphas = lambda_values1, scoring = "neg_mean_squared_error", cv 	= 10, normalize = True)
ridgecv.fit(X_train_P, Y_train_P)
ridgecv
ridgecv.alpha_


ridgecv = RidgeCV(alphas = lambda_values_2, scoring = "neg_mean_squared_error", cv 	= 10, normalize = True)
ridgecv.fit(X_train_P, Y_train_P)
ridgecv
ridgecv.alpha_  #optimum parmeter değeri

#final modeli

ridge_tuned = Ridge(alpha = ridgecv.alpha_).fit(X_train_P, Y_train_P)
y_pred_ridge_p =ridge_tuned.predict(X_test_P)
np.sqrt(mean_squared_error(Y_test_P, y_pred_ridge_p))




#yapay Sinir Ağları start...

#standardscaler() veri setindeki tüm özniteliklerin benzer ölçeklere sahip olması sağlanır
sclar = StandardScaler()
sclar.fit(X_train_P)
X_train_scaled= sclar.transform(X_train_P)
X_test_scaled = sclar.transform(X_test_P)
X_train_scaled
#temel model
mlp_model = MLPRegressor().fit(X_train_scaled ,Y_train_P)
mlp_model
 #tahmin
mlp_model.predict(X_train_scaled)[0:5]
y_pred_scaled_P = mlp_model.predict(X_test_scaled)

#hata oranı hesablama
np.sqrt(mean_squared_error(Y_test_P, y_pred_scaled_P))

#model tunning
#model optemize
#iki tane gizili katman ve bu katman neron sayısı (10 ve 2)  ve (45 ,45) her alpha değeri değerlendirme yaparız
mlp_params = {"alpha" : [ 0.01, 0.2, 0.001, 0.0001],  "hidden_layer_sizes" : [(10,2) ,(45,45)]}
mlp_cv_model = GridSearchCV(mlp_model , mlp_params, cv=5,verbose= 2, n_jobs= 1).fit(X_train_P, Y_train_P)

# model perfomansı artırmak için mlp_params içinde en iyi parametreler bulmak
mlp_cv_model.best_params_
#final utn edilmiş model fit etmek
tuned_mlp_model =  MLPRegressor(alpha=0.02 ,hidden_layer_sizes=(10,2)).fit(X_train_scaled, Y_train_P)
tuned_mlp_model.predict(X_test_scaled)
y_pred_scaled_tuned= tuned_mlp_model.predict(X_test_scaled)
np.sqrt(mean_squared_error(Y_test_P, y_pred_scaled_tuned))