# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:54:24 2021

@author: Nikola
"""

#%% import biblioteka

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets


#%% Ucitavanje DataFrame i provera kako izgleda prvih nekoliko vrsta u bazi

df = pd.read_csv("C:/Users/Nikola/Desktop/Prep oblika/Domaci 1/BeijingPM20100101_20151231.csv")

df.shape
print("Broj obelezja u bazi je: ", df.shape[1] )
print("Broj uzoraka u bazi je: ", df.shape[0] )
print("Sva obelezja u bazi su: ", df.columns)
print("Sva kategoricka obelezja u bazi su: ", df.columns[5],df.columns[0],", ",df.columns[1],", ",df.columns[2],", ",df.columns[3],", ",df.columns[4],", ",df.columns[14] )
print("Sva numericka obelezja u bazi su:", ", ",df.columns[6],", ",df.columns[7],", ",df.columns[8],", ",df.columns[9],", ",df.columns[11],", ",df.columns[12],", ",df.columns[13] )
obelezja = df.columns

dfHead =df.head()
dfTail = df.tail()
df.info()


#%% nedostajuci podaci


numOfIsNull = df.isnull().sum()
sortedNumOfIsNull= df.isnull().sum().sort_values(ascending=False)

percentOfIsNull = df.isnull().sum()/len(df)*100
sortedPercentOfIsNull = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)

missing_data = pd.concat([numOfIsNull, percentOfIsNull], axis=1, keys=['Total', 'Percent'])
missing_data_sorted = pd.concat([sortedNumOfIsNull, sortedPercentOfIsNull], axis=1, keys=['Total', 'Percent'])

#%% izbacivanje obeleznja 'PM Dongsi', 'PM Donqsiguan', 'PM Nonqzhanquan'

df.drop(['PM_Dongsi','PM_Dongsihuan', 'PM_Nongzhanguan'], inplace= True, axis = 1)

#%% izbacivanje obeleznja 'NO', jer predstavlja redni broj uzorka, sto nam ukazuje da to nije relevantan podatak u rezultatima

df.drop('No', inplace= True, axis = 1)

#%%brisanje nan za TEMP. S obzirom da je procenat vrednosti 0 u koloni TEMP < od 0.001 % mozemo ih jednostavno dropovati.
# Dropovanjem uzoraka koje imaju vrednosti 0 u koloni 'TEMP' direktno smo obrisali, uzorke koje takodje sadrze vrednost
# 0 u kolonama cbwd, lws

tempNan = df.loc[df['TEMP'].isnull()]
removableIndexesTemp=df.loc[df['TEMP'].isnull()].index
df.drop(removableIndexesTemp, inplace= True, axis = 0)

#%% brisanje nan za PRES i HUMI. Isti postupak ovde. Brisanjem uzoraka PRES brisemo i uzorke kolone HUMI sa vrednosti 0 

presNan =  df.loc[df['PRES'].isnull()]
removableIndexesPres=df.loc[df['PRES'].isnull()].index
df.drop(removableIndexesPres, inplace= True, axis = 0)

#%% brisanje nan za precipitation i Iprec. Isti postupak ovde. Brisanjem uzoraka precipitation brisemo i uzorke kolone Iprec sa vrednosti 0 

presNan =  df.loc[df['precipitation'].isnull()]
removableIndexesPrecipitation=df.loc[df['precipitation'].isnull()].index
df.drop(removableIndexesPrecipitation, inplace= True, axis = 0)

#drugi nacin za popunjavanje nedostajucih podataka
#df['precipitation'].fillna(df['precipitation'].median(), inplace=True)
#%% #brisanje nan za PM_us POST. S obzirom da nam nedostaje

df['PM_US Post'].fillna(method='bfill',inplace=True)


#%%

print(df.isnull().sum())
df.describe()

#%% Prebacivanje kateg. vrednosti u numer. 

df.loc[df['cbwd']=='cv','cbwd']=0
df.loc[df['cbwd']=='NW','cbwd']=1
df.loc[df['cbwd']=='NE','cbwd']=2
df.loc[df['cbwd']=='SE','cbwd']=3

print(df['cbwd'].unique())


#%% Analiza obelezja(vrsta obeležja, osnovne statistike, raspodela)
#Empirijski opseg

for i in df.columns:
    empiricalScopeR = df[i].max()- df[i].min()
    print("Empirijski opseg r za obeležje",i,"je",empiricalScopeR)



#%% Koeficienti spljostenosti
print("Koeficijenti spljoštenosti:")
for i in df.columns:
    coefficienfOfKurtosis= df[i].kurtosis(axis = 0) 
    print(i , ":" , coefficienfOfKurtosis )
print("\n")

#%% Koeficienti asimetrije
print("Koeficijenti asimetrije:")
for i in df.columns:
    coefficienfOfSkewness= df[i].skew(axis = 0) 
    print(i , ":" , coefficienfOfSkewness )

#%% Korelacije izmedju obelezja

corr = df[["year","month","day","hour","season","PM_US Post","DEWP","HUMI","PRES","TEMP","Iws","precipitation","Iprec"]].corr()
f = plt.figure(figsize=(13, 10))
sns.heatmap(corr.abs(), annot=True);

#%% Matrice korelacije za neke od obelezja

matrica_korelacije = df.corr() 
print(matrica_korelacije['TEMP'])
print(matrica_korelacije['precipitation'])
print(matrica_korelacije['HUMI'])
print(matrica_korelacije['PRES'])
print(matrica_korelacije['PM_US Post'])

#%%iscrtavanje nekih od korelacija

fig, axs = plt.subplots(3,3,figsize=(12,12))
axs[0, 0].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'TEMP'], color='black',label="PM_US Post-TEMP")
axs[0, 0].set_title('PM_US Post - TEMP')
axs[0, 1].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'PRES'], color='black',label="PM_US Post-PRES")
axs[0, 1].set_title('PM_US Post - PRES')
axs[0, 2].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'HUMI'], color='black',label="PM_US Post-HUMI")
axs[0, 2].set_title('PM_US Post - HUMI')
axs[1, 0].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'season'], color='blue', label="PM_US Post-season")
axs[1, 0].set_title('PM_US Post - season')
axs[1, 1].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'Iws'], color='blue', label="PM_US Post-Iws")
axs[1, 1].set_title('PM_US Post - Iws')
axs[1, 2].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'month'], color='blue', label="PM_US Post-month")
axs[1, 2].set_title('PM_US Post  - month')
axs[2, 0].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'precipitation'], color='red', label="PM_US Post-precipitation")
axs[2, 0].set_title('PM_US Post - precipitation')
axs[2, 1].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'day'], color='red', label="PM_US Post-day")
axs[2, 1].set_title('PM_US Post - day')
axs[2, 2].scatter(df.loc[:, 'PM_US Post'],df.loc[:, 'hour'], color='red', label="'PM_US Post-hour")
axs[2, 2].set_title('PM_US Post - hour')


#%%
df['PM_US Post'].describe()
plt.boxplot([df.loc[:, 'PM_US Post']]) 
plt.grid()

#%%GODINE

df['year'].unique()

df2010 = df[df['year']==2010]
df2011 = df[df['year']==2011]
df2012 = df[df['year']==2012]
df2013 = df[df['year']==2013]
df2014 = df[df['year']==2014]
df2015 = df[df['year']==2015]

df2010 = df2010.set_index('year')
df2011 = df2011.set_index('year')
df2012 = df2012.set_index('year')
df2013 = df2013.set_index('year')
df2014 = df2014.set_index('year')
df2015 = df2015.set_index('year')

fig, axs = plt.subplots(2,3,figsize=(9,9))
axs[0, 0].boxplot([df2010.loc[:, 'PM_US Post']]) 
axs[0, 0].set_title('2010')
axs[0, 0].grid()

axs[0, 1].boxplot([df2011.loc[:, 'PM_US Post']]) 
axs[0, 1].set_title('2011')
axs[0, 1].grid()

axs[0, 2].boxplot([df2012.loc[:, 'PM_US Post']]) 
axs[0, 2].set_title('2012')
axs[0, 2].grid()

axs[1, 0].boxplot([df2013.loc[:, 'PM_US Post']]) 
axs[1, 0].set_title('2013')
axs[1, 0].grid()

axs[1, 1].boxplot([df2014.loc[:, 'PM_US Post']]) 
axs[1, 1].set_title('2014')
axs[1, 1].grid()

axs[1, 2].boxplot([df2015.loc[:, 'PM_US Post']]) 
axs[1, 2].set_title('2015')
axs[1, 2].grid()


df2010.describe()
df2011.describe()
df2012.describe()
df2013.describe()
df2014.describe()
df2015.describe()

#%% Meseci pretvoreni u sezone

df['season'].unique()

dfW1= df.loc[df['month']==1]
dfW2= df.loc[df['month']==2]   
dfW3= df.loc[df['month']==12]
dfWinter = pd.concat([dfW1,dfW2,dfW3])

dfSp1= df.loc[df['month']==3]
dfSp2= df.loc[df['month']==4]   
dfSp3= df.loc[df['month']==5]
dfSpring = pd.concat([dfSp1,dfSp2,dfSp3])

dfSu1= df.loc[df['month']==6]
dfSu2= df.loc[df['month']==7]   
dfSu3= df.loc[df['month']==8]
dfSummer = pd.concat([dfSu1,dfSu2,dfSu3])

dfA1= df.loc[df['month']==9]
dfA2= df.loc[df['month']==10]   
dfA3= df.loc[df['month']==11]
dfAutumn = pd.concat([dfA1,dfA2,dfA3])

fig, axs = plt.subplots(2,2,figsize=(9,8))
axs[0, 0].boxplot([dfWinter.loc[:, 'PM_US Post']]) 
axs[0, 0].set_title('WINTER')
axs[0, 0].grid()
dfWinter.describe()

axs[0, 1].boxplot([dfSpring.loc[:, 'PM_US Post']]) 
axs[0, 1].set_title('SPRING')
axs[0, 1].grid()
dfSpring.describe()

axs[1, 0].boxplot([dfSummer.loc[:, 'PM_US Post']]) 
axs[1, 0].set_title('SUMMER')
axs[1, 0].grid()
dfSummer.describe()

axs[1, 1].boxplot([dfAutumn.loc[:, 'PM_US Post']]) 
axs[1, 1].set_title('AUTUMN')
axs[1, 1].grid()
dfAutumn.describe()


#%% Neka proizvoljda vrsta analize


#koliko razlicitih godina,meseci, dana, sezona da bi se videlo da li ima slucajno nekih gresaka u unosu vrednosti

print('Godine: \n', df['year'].unique())
print('\n Meseci: \n', df['month'].unique())
print('\n Dani: \n', df['day'].unique())
print('\n Sezone: \n', df['season'].unique())


num1 = df.loc[(df["season"]==1),: ]
num2= df.loc[(df["season"]==2),: ]
num3= df.loc[(df["season"]==3),: ]
num4= df.loc[(df["season"]==4),: ]



avgTemperature= df.groupby("season")["TEMP"].mean()
avgPost= df.groupby("season")["PM_US Post"].mean()

avgTemperatureByMonth= df.groupby("month")["TEMP"].mean()
avgPostByMonth = df.groupby("month")["PM_US Post"].mean()

#%% Prosecno PM_US Post po mesecima u svakoj od godini

avgPostByMonth2010 = df.loc[df['year'] ==2010].groupby("month")["PM_US Post"].mean()
avgPostByMonth2011 = df.loc[df['year'] ==2011].groupby("month")["PM_US Post"].mean()
avgPostByMonth2012 = df.loc[df['year'] ==2012].groupby("month")["PM_US Post"].mean()
avgPostByMonth2013 = df.loc[df['year'] ==2013].groupby("month")["PM_US Post"].mean()
avgPostByMonth2014 = df.loc[df['year'] ==2014].groupby("month")["PM_US Post"].mean()
avgPostByMonth2015 = df.loc[df['year'] ==2015].groupby("month")["PM_US Post"].mean()

mylabels= ['2010','2011','2012','2013','2014','2015']
plt.figure(figsize=(8, 6), dpi=80)
avgPostByMonth2010.plot()
avgPostByMonth2011.plot()
avgPostByMonth2012.plot()
avgPostByMonth2013.plot()
avgPostByMonth2014.plot()
avgPostByMonth2015.plot()
plt.xlabel('Month')
plt.ylabel('PM_US Post')
plt.legend(labels=mylabels)
plt.grid()

#%%

mylabels2 = ['HUMI over 50', 'HUMI under 50']
HUMImore = df.loc[df['HUMI'] >50].groupby("year")["PM_US Post"].mean().plot()

HUMIless = df.loc[df['HUMI'] <50].groupby("year")["PM_US Post"].mean().plot()
plt.ylabel('Year')
plt.ylabel('PM_US Post')
plt.legend(labels=mylabels2)
#%%

df['Iws'].max()

mylabels= ['0-50','50-100','100-150','150-200','200-250','250-300','300-350','350-400','400-450','450-500','500-550','550-600']
plt.figure(figsize=(9, 6), dpi=80)
lwsPOST50 = df.loc[(df['Iws']>0) & (df['Iws']<50)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST100 = df.loc[(df['Iws']>50) & (df['Iws']<1000)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST150 = df.loc[(df['Iws']>100) & (df['Iws']<150)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST200 = df.loc[(df['Iws']>150) & (df['Iws']<200)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST250 = df.loc[(df['Iws']>200) & (df['Iws']<250)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST300 = df.loc[(df['Iws']>250) & (df['Iws']<300)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST350 = df.loc[(df['Iws']>300) & (df['Iws']<350)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST400 = df.loc[(df['Iws']>350) & (df['Iws']<400)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST450 = df.loc[(df['Iws']>400) & (df['Iws']<450)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST500 = df.loc[(df['Iws']>450) & (df['Iws']<500)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST550 = df.loc[(df['Iws']>500) & (df['Iws']<550)].groupby("year")["PM_US Post"].mean().plot()
lwsPOST650 = df.loc[(df['Iws']>5500) & (df['Iws']<600)].groupby("year")["PM_US Post"].mean().plot()
plt.xlabel('YEAR')
plt.ylabel('PM_US Post')
plt.legend(labels=mylabels)
plt.grid()



#%%
mylabels=['North-West','cv','North-East','South-East']
plt.figure(figsize=(9, 6), dpi=80)
cbwdNWPOST = df.loc[df['cbwd']==1 ].groupby("year")["PM_US Post"].mean().plot()
cbwdCVPOST = df.loc[df['cbwd']==0 ].groupby("year")["PM_US Post"].mean().plot()
cbwdNEPOST = df.loc[df['cbwd']==2 ].groupby("year")["PM_US Post"].mean().plot()
cbwdSEPOST = df.loc[df['cbwd']==3].groupby("year")["PM_US Post"].mean().plot()
plt.xlabel('YEAR')
plt.ylabel('PM_US Post')
plt.legend(labels=mylabels)
plt.grid()

#%%

plot= df[["PM_US Post","month"]].groupby("month").median().reset_index()
sns.pointplot(x="month", y="PM_US Post", data=plot)


#%%



noRainPOST = df.loc[df['precipitation']==0 ]
RainPOST = df.loc[df['precipitation']!=0 ]

plt.figure(figsize=(8, 6), dpi=80)
plt.hist(noRainPOST.loc[:,'PM_US Post'] , density=True, alpha=0.3, bins=50, label ='Sa kisom',range=(0,500))
plt.hist(RainPOST.loc[:,'PM_US Post'] , density=True, alpha=0.3, bins=50, label ='Bez kise',range=(0,500))
plt.title("Odnos PM cestica u vazduhu sa kisom i bez")
plt.xlabel("PM_US Post")
plt.ylabel("Verovatnoca")
plt.legend()

#%%

#%% linearna regresija

print(df.isnull().sum().sum())
statistical_analysis = df.describe()

#%% u nasem dataFrame-u uocavamo kategoricka obelezja koja bi trebali da odbacimo pre nego sto krenemo 
# sa racunanjem linearne regresije
x = df.copy()
x.drop(['year','month', 'day','hour','season','PM_US Post'], inplace= True, axis = 1)
y=df['PM_US Post']

print(x.shape)
print(x.columns)
x.head()

#%%
statistical_analysisX = x.describe()
statistical_analysisX = y.describe()

#%%
plt.figure(figsize=(10,5))
plt.hist(y, density=True, bins=20)
plt.show()

#%%

class LinearRegressionUsingGD:
    """
    eta : Brzina ucenja
    n_iterations : Broj iteracija algoritma
    
    w_ : koeficijenti obucenog modela
    cost_ : vrednost funkcije cene
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """
        Ulazni parametri
        -------
        x : numpy array, shape = [n_samples, n_features]
            Trening uzorci
        y : numpy array, shape = [n_samples, n_target_values]
            Ciljne vrednosti izlaza
            
        Izlazni parametri
        -------
        self : object (obučen model)
        """

        self.cost_ = []
        x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)  #Prosirivanje obelezja za jos jednu kolonu jedinica da bi se zadovoljila dimenzionalnost
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0] #broj uzoraka

        for _ in range(self.n_iterations):  #ideja da se kroz sveisteracije prodje i vrti formula za racunanje cene, gradienta, i abdejtovanje tezina
            y_pred = np.dot(x, self.w_)   # Predikcije za tezine  7.2 formula
            residuals = y_pred - y      # razlika izmedju prediktovane i ciljne vrednosti
            gradient_vector = np.dot(x.T, residuals)   #7.4
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self


    def predict(self, x):
        """ 
        Ulazni parametri
        ----------
        x : numpy array, shape = [n_samples, n_features]
            Test uzorci
        Izlazni parametri
        -------
        Predviđene vrednosti zavisne promenljive
        """
        x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1) #prosirujemo sa kolonom jedinica
        
        return np.dot(x, self.w_)  #predikcija sa mnozenjem


def model_evaluation(y, y_predicted, N, d):  #MERE ZA USPESNOST REGRESORA (y -ciljne, y_predicted - prediktovane , N-broj uzoraka koji su dostupni, d-br. obelezja)  
                                 
    mse = mean_squared_error(y_test, y_predicted)  #SQUERD ERROR  -- nam govori koliko su nase predvidjene vrednosti blizu linije regresije,
                                                    #predstavlja razliku izmedju stvarnih i predvidjenih vrednosti dignutu na kvadrat
                                                    
                                                    
    mae = mean_absolute_error(y_test, y_predicted)  #APSOLUT ERROR  -- vise kaznjava outliere
    
    rmse = np.sqrt(mse)                             #koren iz MSE, kaznjava lose predikcije
    
    
    r2 = r2_score(y_test, y_predicted)              #govori koliki udeo varijanse pokriva model odnosno koliki procenat
                                                    #varijanse moze biti objasnjen modelom,sto je vise to je bolje
                                                    
                                                    
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)              #uzima u obzir i broj obelezja jer bi poredjenje modela
                                                #s razlicitim brojem obelezja bilo nemoguce

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)                    #Mera uspesnosti regresora koliko mi gresimo s predikcijama u odnosu na predikciju (blizu 1 je dobra predikcija)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))




#%% podela skupa na trening i test podatke
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)    #delimo skup (x- ulazi, y-ciljne vrednosti, test_size- stavljamo po strani)

#%%
# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)  #FIT_INTERCEPT - da li cemo da imamo slobodan clan ili ne (true - imamo slobodan clan)

# Obuka
first_regression_model.fit(x_train, y_train)  #klasa metoda fit

# Testiranje
y_predicted = first_regression_model.predict(x_test) #predikcije

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)


#DA LI SU NAM SVA OBELEZJA POTREBNA? --REDUKCIJA OBELEZJA --- SELEKCIJA UNAPRED / SELEKCIJA UNAZAD
#SELEKCIJA UNAPRED - ZA SVAKO OBELEZJE JEDAN MODEL
#SELEKCIJA UNAZAD - KREIRA MODEL SA SVIM OBELEZJIMA I ONDA GLEDA NEKE STATISTICKE VREDNOSTI PO SVAKOM OBELEZJU

#%%


scaler = StandardScaler()   #Standardizacija
scaler.fit(x_train)         #nije klasicno ucenje, izvlace srednju vrednost i varijansu
x_train_std = scaler.transform(x_train)    #Standardizacija, pa su oni uporedivi
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std.head()


#%% Selekcija obelezja

X = sm.add_constant(x_train)
X2 = sm.add_constant(x_train)
X.set_index(X2.index)
X = np.array(X)
y_train = np.array(y_train)

#%%
                                                            #OLS ORDER LISTS SQUERS minimizuje srednjekvadratnu gresku
model = sm.OLS(y_train, X.astype('float')).fit()   #gde se na ovaj nacin testiraju sva obelezja (GLEDAMO P VREDNOST - ona posmatra prag npr 1 %
model.summary()                                    #tj, 0,01 i gde god je ona manja od te vrednosti mi bi trebali da zadrzimo to obelezje)

sns.distplot(model.resid, fit=stats.norm);

#%%

#DOVODI DO BRZE KONVERGENCIJE KA OPTIMALNOM RESENJU


regression_model_std = LinearRegression()

# Obuka modela
regression_model_std.fit(x_train_std, y_train)

# Testiranje
y_predicted = regression_model_std.predict(x_test_std)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.show()
print("koeficijenti: ", regression_model_std.coef_)

corr = x.corr()
f = plt.figure(figsize=(12, 9))
sns.heatmap(corr.abs(), annot=True);


#%% druge hipotez

#-ideja je da se prosiri osnovna hipoteza, da model nije sastavljen samo kao linearna kombinacija vec mesavina obelezja
#mana: overfitting
#pozeljan je da ne bude preobucen a da moze da resava problem


                                                                        #za difoltnu vrednost postavlja 2 stepen
poly = PolynomialFeatures(interaction_only=True, include_bias=False)    
                                                                        #interaction_only - osim osnovnih obelezja, posmatrace i interakcije izmedju njih
                                                                        #True daje samo interakciju a False i kvadrate
                                                    
x_inter_train = poly.fit_transform(x_train_std)                         #include_bias - false jer ce polynomialFeatures ce da napravi automacki slobodan clan
x_inter_test = poly.transform(x_test_std)                               #ako je true uglavnom ce biti nula vrednost

print(poly.get_feature_names())                                         #fit transform fituje pa transformise u odredjenu promenljivu, transform od postojecih 
                                                                            #obelezja dobija neka nova

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()   #klasa pravi sam svoj slobodan clan pa je najbolje gore da se ne kreira

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)

#%%

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
plt.show()
print("koeficijenti: ", regression_model_degree.coef_)

#%%Ridge regresija

# Inicijalizacija
ridge_model = Ridge(alpha=5)      #alpha stepen kaznavanja onih odstupanja

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)
#%%Lasso regresija RIDGE I LASSO

  #sluze za borbu protiv natprelagodjavanja (ako se ona nauci na nekom setu podataka, nece biti dobra za neke druge setove)      

#-RIDGE minimizuje L2 normu
#-LASSO minimizuje L1 normu, moze da podesi neke koeficijente na 0 odnosno izbaci obelezja
#-opredeljujemo se za onaj koji ima manje koeficijente



lasso_model = Lasso(alpha=0.01) #-alpha je stepen s kojim penalizujemo visoke tezine

# Fit the data(train the model)
lasso_model.fit(x_inter_train, y_train)

# Predict
y_predicted = lasso_model.predict(x_inter_test)

# Evaluation
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)


#%%
plt.figure(figsize=(10,5))
plt.plot(regression_model_degree.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'linear',zorder=7) # zorder for ordering the markers
plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') # alpha here is for transparency
plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='best')
plt.show()
