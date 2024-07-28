# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:11:13 2024

@author: leoim
"""

import pandas as pd
import getpass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

user = getpass.getuser()

df = pd.read_csv('C:/Users/'+user+'/Downloads/Delivery.csv', delimiter=',')

########### LIMPANDO A BASE DE VALORES NEGATIVOS, NULOS E DUPLICADOS ############

columns_to_clean = ['awaited_time', 'store_primary_category', 'order_protocol', 'total_items', 'subtotal', 'num_distinct_items',
'min_item_price', 'max_item_price', 'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders'
, 'estimated_store_to_consumer_driving_duration']
valoresNegativos = []

for column in columns_to_clean:
  for a in range(len(df)):
    if df[column].iloc[a] <= 0:
      valoresNegativos.append(a)
      
df = df.drop(valoresNegativos, axis = 'index')

df.dropna(inplace=True)
df = df.drop_duplicates()

########### REMOVENDO OS OUTLIERS ############

threshold = 3

for column in columns_to_clean:
  q1 = df[column].quantile(0.25)
  q3 = df[column].quantile(0.75)
  iqr = q3 - q1

  upper_bound = q3 + threshold * iqr
  lower_bound = q1 - threshold * iqr

  df = df[((df[column] <= upper_bound) & (df[column] >= lower_bound))]
  
########### LIMPANDO OS DELIVERYS ALEATÓRIOS ############
  
lista_drop = []
  
for i in range(len(df)):
    if df['delivery_location'].iloc[i] == 'casa do davi' or df['delivery_location'].iloc[i] == 'casa do sasaki' or df['delivery_location'].iloc[i] == 'rei das batidas':
        lista_drop.append(i)
        
df = df.drop(lista_drop)
df = df.drop([47310, 4726, 62821, 97665, 99515], axis=0)

########### ARRUMANDO AS COLUNAS DE DATA E HORA ############

df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

# Coluna 'Datacreated'
df['mounth_created'] = df['created_at'].dt.strftime('%m')
df['hora_created'] = df['created_at'].dt.strftime('%H')

# Coluna 'Datacreated'
df['mounth_delivery'] = df['actual_delivery_time'].dt.strftime('%m')
df['hora_delivery'] = df['actual_delivery_time'].dt.strftime('%H')

del df['created_at']
del df['actual_delivery_time']

########### AGRUPAMENTO ############

df_agrup_location = df.groupby(by='delivery_location').agg('sum')

# LABEL ENCODER PRA TRANSFORMAR A STRING EM INT ######################
le = LabelEncoder()
df['delivery_location'] = le.fit_transform(df['delivery_location'])
######################################################################

df_agrup_mi = df.groupby(by='market_id').agg('sum')

########### ANALISE INICIAL ############

df_describe = df.describe()
df_describe.loc['amp'] = df_describe.loc['max'] - df_describe.loc['min']
df_describe.loc['cv'] = (df_describe.loc['std'] / df_describe.loc['mean']).abs() * 100
df_describe.loc['skewness'] = df.skew()
df_describe.round(2)

########### COEFICIENTES DE CORRELAÇÃO ############

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2f')
plt.show()

########### REMOVENDO AS COLUNAS QUE NÃO TEM TANTA CORRELAÇÃO COM AWAITED_TIME ############

df_predicao = df.drop(['market_id', 'store_primary_category', 'min_item_price', 'mounth_created', 'mounth_delivery'], axis=1)

########### DIVIDINDO A BASE PARA FAZER A PREDIÇÃO DE AWAITED ############

X = df_predicao.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,12]]
y = df_predicao.iloc[:,10]

########### PADRONIZANDO A BASE ############

scaler = StandardScaler()
X = scaler.fit_transform(X)

########### DIVIDINDO ENTRE TESTE E TREINAMENTO ############

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.2, stratify = y)

########### EXPORTAR PARA FAZER A ANALISE NO POWER BI  ############

df.to_excel('database_restaurante.xlsx')

########### FAZENDO A PREDIÇÃO E ANALISE ############

resultados_svm = []
resultados_rede_neural = []

def mse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_squared_error(y, y_pred)

def r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return r2_score(y, y_pred)

mse = make_scorer(mse_scorer, greater_is_better=False)
r2 = make_scorer(r2_scorer)

for i in range(10):
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    
    svm = SVC(kernel='rbf', C=2.0)
    scores_svm = cross_val_score(svm, X, y, cv=kfold, scoring='accuracy')
    mse_scores_svm = cross_val_score(svm, X, y, cv=kfold, scoring=mse)
    r2_scores_svm = cross_val_score(svm, X, y, cv=kfold, scoring=r2)
    
    resultados_svm.append({
        'accuracy': scores_svm.mean(),
        'mse': -mse_scores_svm.mean(),
        'r2': r2_scores_svm.mean()
    })

    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')
    scores_rn = cross_val_score(rede_neural, X, y, cv=kfold, scoring='accuracy')
    mse_scores_rn = cross_val_score(rede_neural, X, y, cv=kfold, scoring=mse)
    r2_scores_rn = cross_val_score(rede_neural, X, y, cv=kfold, scoring=r2)
    
    resultados_rede_neural.append({
        'accuracy': scores_rn.mean(),
        'mse': -mse_scores_rn.mean(),  
        'r2': r2_scores_rn.mean()
    })

resultados_svm_df = pd.DataFrame(resultados_svm)
resultados_rede_neural_df = pd.DataFrame(resultados_rede_neural)

print("Resultados do SVM:")
print(resultados_svm_df.describe())
print("\nResultados da Rede Neural:")
print(resultados_rede_neural_df.describe())  