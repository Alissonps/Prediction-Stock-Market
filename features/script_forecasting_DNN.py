#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#--- Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import forecasting_util as fu
import multiprocessing
import warnings

#from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from keras.models import Sequential
from keras.layers import Dense

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA

class Features(object):
    def __init__(self, train_OHCLV, train_anomalous, train_autoencoder, train_rbm, train_pca, 
                 train_OHCLV_pca, train_OHCLV_autoencoder, train_OHCLV_rbm, train_OHCLV_anomalous, 
                 train_pca_autoencoder, train_pca_rmb, train_pca_anomalous, train_rbm_autoencoder, 
                 train_rbm_anomalous, train_autoencoder_anomalous, target_returns, target_volatility):
        self._train_OHCLV = train_OHCLV
        self._train_anomalous = train_anomalous
        self._train_autoencoder = train_autoencoder
        self._train_rbm = train_rbm
        self._train_pca = train_pca
        self._train_OHCLV_pca = train_OHCLV_pca
        self._train_OHCLV_autoencoder = train_OHCLV_autoencoder
        self._train_OHCLV_rbm = train_OHCLV_rbm
        self._train_OHCLV_anomalous = train_OHCLV_anomalous
        self._train_pca_autoencoder = train_pca_autoencoder
        self._train_pca_rbm = train_pca_rmb
        self._train_pca_anomalous = train_pca_anomalous
        self._train_rbm_autoencoder = train_autoencoder
        self._train_rbm_anomalous = train_rbm_anomalous
        self._train_autoencoder_anomalous = train_autoencoder_anomalous

        self._target_returns = target_returns
        self._target_volatility = target_volatility
        
    def __train_OHCLV(self):
        return self._train_OHCLV

    def __train_anomalous(self):
        return self._train_anomalous

    def __train_autoencoder(self):
        return self._train_autoencoder

    def __train_rbm(self):
        return self._train_rbm

    def __train_pca(self):
        return self._train_pca

    def __target_returns(self):
        return self._target_returns

    def __target_volatility(self):
        return self._target_volatility

    def __train_OHCLV_pca(self):
        return self._train_OHCLV_pca
    
    def __train_OHCLV_autoencoder(self):
        return self._train_OHCLV_autoencoder
    
    def __train_OHCLV_rbm(self):
        return self._train_OHCLV_rbm
    
    def __train_OHCLV_anomalous(self):
        return self._train_OHCLV_anomalous
    
    def __train_pca_autoencoder(self):
        return self._train_pca_autoencoder
    
    def __train_pca_rmb(self):
        return self._train_pca_rmb
    
    def __train_pca_anomalous(self):
        return self._train_pca_anomalous
    
    def __train_rbm_autoencoder(self):
        return self._train_rbm_autoencoder
    
    def __train_rbm_anomalous(self):
        return self._train_rbm_anomalous
    
    def __train_autoencoder_anomalous(self):
        return self._train_autoencoder_anomalous

def load_dataset(asset):

    #--- Carrega a série original    
    #home_dir = os.getenv("HOME")
    print (' -> Loading asset file: %s' % asset),
    df_train = pd.read_csv('./data/' + asset)
    print (' -> Done.')
    print (' -> Original dataset dimensions: {} samples, with {} attributes.'.format(df_train.shape[0], df_train.shape[1]))
    return df_train

def load_features_files(asset):

    train_OHCLV = np.loadtxt('./data/001_' + asset, delimiter=',') # Open, High, Low, Close, Volume
    train_anomalous = np.loadtxt('./data/002_' + asset, delimiter=',') # Features extraídas do pacote R de Rob Hyndman
    train_autoencoder = np.loadtxt('./data/003_' + asset, delimiter=',') # Features extraídas através de um autoencoder
    train_rbm = np.loadtxt('./data/004_' + asset, delimiter=',') # Features extraídas com uma RBM
    train_pca = np.loadtxt('./data/005_' + asset, delimiter=',') # Features extraídas com PCA

    target_returns = np.loadtxt('./data/t1_' + asset, delimiter=',') # Target 1: Retorno do próximo dia
    target_volatility = np.loadtxt('./data/t2_' + asset, delimiter=',') # Target 2: Volatilidade do próximo dia

    dtf_train_OHCLV = pd.DataFrame(train_OHCLV)
    dtf_train_pca = pd.DataFrame(train_pca)
    dtf_train_rbm = pd.DataFrame(train_rbm)
    dtf_train_autoencoder = pd.DataFrame(train_autoencoder)
    dtf_train_anomalous = pd.DataFrame(train_anomalous)
    
    dtf_train_OHCLV_pca = pd.concat([dtf_train_OHCLV, dtf_train_pca], axis=1)
    dtf_train_OHCLV_autoencoder = pd.concat([dtf_train_OHCLV, dtf_train_autoencoder], axis=1)
    dtf_train_OHCLV_rbm = pd.concat([dtf_train_OHCLV, dtf_train_rbm], axis=1)
    dtf_train_OHCLV_anomalous = pd.concat([dtf_train_OHCLV, dtf_train_anomalous], axis=1)
    dtf_train_pca_autoencoder = pd.concat([dtf_train_pca, dtf_train_autoencoder], axis=1)
    dtf_train_pca_rbm = pd.concat([dtf_train_pca, dtf_train_rbm], axis=1)
    dtf_train_pca_anomalous = pd.concat([dtf_train_pca, dtf_train_anomalous], axis=1)
    dtf_train_rbm_autoencoder = pd.concat([dtf_train_rbm, dtf_train_autoencoder], axis=1)
    dtf_train_rbm_anomalous = pd.concat([dtf_train_rbm, dtf_train_anomalous], axis=1)
    dtf_train_autoencoder_anomalous = pd.concat([dtf_train_autoencoder, dtf_train_anomalous], axis=1)
    
    train_OHCLV_pca = dtf_train_OHCLV_pca.values
    train_OHCLV_autoencoder = dtf_train_OHCLV_autoencoder.values
    train_OHCLV_anomalous = dtf_train_OHCLV_anomalous.values
    train_OHCLV_rbm = dtf_train_OHCLV_rbm.values
    train_pca_anomalous = dtf_train_pca_anomalous.values
    train_pca_autoencoder = dtf_train_pca_autoencoder.values
    train_pca_rbm = dtf_train_pca_rbm.values
    train_rbm_autoencoder = dtf_train_rbm_autoencoder.values
    train_rbm_anomalous = dtf_train_rbm_anomalous.values
    train_autoencoder_anomalous = dtf_train_autoencoder_anomalous

    df = Features(train_OHCLV=train_OHCLV, train_anomalous=train_anomalous, train_autoencoder=train_autoencoder,
                  train_rbm=train_rbm, train_pca=train_pca, train_OHCLV_pca=train_OHCLV_pca, train_OHCLV_autoencoder=train_OHCLV_autoencoder, train_OHCLV_rbm=train_OHCLV_rbm, train_OHCLV_anomalous=train_OHCLV_anomalous, 
                  train_pca_autoencoder=train_pca_autoencoder, train_pca_rmb=train_pca_rbm, train_pca_anomalous=train_pca_anomalous, train_rbm_autoencoder=train_rbm_autoencoder, 
                  train_rbm_anomalous=train_rbm_anomalous, train_autoencoder_anomalous=train_autoencoder_anomalous, target_returns=target_returns, target_volatility=target_volatility)

    return df

warnings.filterwarnings("ignore")
def train_test_arima(df):

    target_returns = df._target_returns
    # target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()
    target_returns = scaler.fit_transform(target_returns)
    # target_volatility = scaler.fit_transform(target_volatility)

    rmse_returns_OHCLV = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target_returns):
        y_train_OHCLV, y_test_OHCLV = np.transpose(target_returns)[train_index], np.transpose(target_returns)[test_index]

        # order = fu.get_PDQ_parallel(y_train_OHCLV, multiprocessing.cpu_count())
        # print ('Estimated order for ARIMA model: {}'.format(order))
        order = (0,0,1)

        model_returns_OHCLV = ARIMA(y_test_OHCLV, order=order)
        results_AR = model_returns_OHCLV.fit(disp=-1)
        pred = results_AR.fittedvalues

        rmse = np.sqrt(mean_squared_error(y_test_OHCLV, pred))
        # print('RMSE - OHCLV: {}'.format(rmse))
        rmse_returns_OHCLV.append(rmse)

    res = np.mean(rmse_returns_OHCLV)
    print(' -> RMSE - ARIMA - Returns: {}'.format(res))

    return res

warnings.filterwarnings("ignore")
def train_test_garch(df):

    target_returns = df._target_returns
    scaler = preprocessing.MinMaxScaler()
    target_returns = scaler.fit_transform(target_returns)
    rmse_returns_OHCLV = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target_returns):
        y_train_OHCLV, y_test_OHCLV = np.transpose(target_returns)[train_index], np.transpose(target_returns)[
            test_index]

        model_returns_OHCLV = arch_model(y_test_OHCLV, vol='Garch', p=1, o=0, q=1, dist='Normal')
        results_GARCH = model_returns_OHCLV.fit(update_freq=10, disp='off')
        pred = results_GARCH.resid

        rmse = np.sqrt(np.mean(np.power(pred,2)))
        rmse_returns_OHCLV.append(rmse)

    res = np.mean(rmse_returns_OHCLV)
    print(' -> RMSE - GARCH(1,1) - Returns: {}'.format(res))

    return res

def get_predictions_dnn(train, target):

    rmse_returns = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target):
        # --- Cross validation - OHCLV

        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = np.transpose(target)[train_index], np.transpose(target)[
            test_index]

        y_train = np.transpose(y_train)
        y_test = np.transpose(y_test)

        model = Sequential()
        model.add(Dense(64, input_dim=(X_train.shape)[1], 
                init='normal', activation='relu'))
        model.add(Dense(64, init='normal', activation='relu'))
        model.add(Dense(32, init='normal', activation='relu'))
        model.add(Dense(32, init='normal', activation='relu'))
        model.add(Dense(16, init='normal', activation='relu'))
        model.add(Dense(16, init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, nb_epoch=100, batch_size=32,
                        validation_data=(X_test, y_test), shuffle=False, verbose=0)
        pred = model.predict(X_test, batch_size=32, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_returns.append(rmse)

    return np.mean(rmse_returns)

warnings.filterwarnings("ignore")
def train_test_dnn(df):
    train_OHCLV = df._train_OHCLV
    train_anomalous = df._train_anomalous
    train_autoencoder = df._train_autoencoder
    train_rbm = df._train_rbm
    train_pca = df._train_pca
    train_OHCLV_pca = df._train_OHCLV_pca
    train_OHCLV_rbm = df._train_OHCLV_rbm
    train_OHCLV_autoencoder = df._train_OHCLV_autoencoder
    train_OHCLV_anomalous = df._train_OHCLV_anomalous
    train_PCA_anomalous = df._train_pca_anomalous
    train_PCA_autoencoder = df._train_pca_autoencoder
    train_PCA_rbm = df._train_pca_rbm

    #target_returns = df._target_returns
    target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()


    train_anomalous = scaler.fit_transform(train_anomalous)
    train_autoencoder = scaler.fit_transform(train_autoencoder)
    train_rbm = scaler.fit_transform(train_rbm)
    train_pca = scaler.fit_transform(train_pca)
    train_OHCLV_anomalous = scaler.fit_transform(train_OHCLV_anomalous)
    train_OHCLV_autoencoder = scaler.fit_transform(train_OHCLV_autoencoder)
    train_OHCLV_pca = scaler.fit_transform(train_OHCLV_pca)
    train_OHCLV_rbm = scaler.fit_transform(train_OHCLV_rbm)
    train_OHCLV = scaler.fit_transform(train_OHCLV)
    train_PCA_anomalous = scaler.fit_transform(train_PCA_anomalous)
    train_PCA_autoencoder = scaler.fit_transform(train_PCA_autoencoder)
    train_PCA_rbm = scaler.fit_transform(train_PCA_rbm)

    #target_returns = scaler.fit_transform(target_returns)
    target_volatility = scaler.fit_transform(target_volatility)

    res_OHCLV = get_predictions_dnn(train_OHCLV, target_volatility)
    res_Anomalous = get_predictions_dnn(train_anomalous, target_volatility)
    res_Autoencoder = get_predictions_dnn(train_autoencoder, target_volatility)
    res_PCA = get_predictions_dnn(train_pca, target_volatility)
    res_RBM = get_predictions_dnn(train_rbm, target_volatility)
    res_OHCLV_Anomalous = get_predictions_dnn(train_OHCLV_anomalous, target_volatility)
    res_OHCLV_Autoencoder = get_predictions_dnn(train_OHCLV_autoencoder, target_volatility)
    res_OHCLV_PCA = get_predictions_dnn(train_OHCLV_pca, target_volatility)
    res_OHCLV_RBM = get_predictions_dnn(train_OHCLV_rbm, target_volatility)
    res_PCA_Anomalous = get_predictions_dnn(train_PCA_anomalous, target_volatility)
    res_PCA_Autoencoder = get_predictions_dnn(train_PCA_autoencoder, target_volatility)
    res_PCA_RBM = get_predictions_dnn(train_PCA_rbm, target_volatility)
    
    return res_OHCLV, res_Anomalous, res_Autoencoder, res_PCA, res_RBM, res_OHCLV_Anomalous, res_OHCLV_Autoencoder, res_OHCLV_PCA, res_OHCLV_RBM, res_PCA_Anomalous, res_PCA_Autoencoder, res_PCA_RBM, 

def get_predictions_rf(train, target):

    rmse_returns = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target):
        # --- Cross validation - OHCLV

        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = np.transpose(target)[train_index], np.transpose(target)[
            test_index]

        y_train = np.transpose(y_train)
        y_test = np.transpose(y_test)

        model = RandomForestRegressor(n_estimators=200).fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_returns.append(rmse)

    return np.mean(rmse_returns)

warnings.filterwarnings("ignore")
def train_test_rf(df):

#     train_OHCLV_pca = df._train_OHCLV_pca
#     train_OHCLV_rbm = df._train_OHCLV_rbm
#     train_OHCLV_autoencoder = df._train_OHCLV_autoencoder
#     train_OHCLV_anomalous = df._train_OHCLV_anomalous
    train_PCA_anomalous = df._train_pca_anomalous
    train_PCA_autoencoder = df._train_pca_autoencoder
    train_PCA_rbm = df._train_pca_rbm
#     train_OHCLV = df._train_OHCLV
#     train_anomalous = df._train_anomalous
#     train_autoencoder = df._train_autoencoder
#     train_rbm = df._train_rbm
#     train_pca = df._train_pca

    target_returns = df._target_returns
    target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()

#     train_OHCLV_anomalous = scaler.fit_transform(train_OHCLV_anomalous)
#     train_OHCLV_autoencoder = scaler.fit_transform(train_OHCLV_autoencoder)
#     train_OHCLV_pca = scaler.fit_transform(train_OHCLV_pca)
#     train_OHCLV_rbm = scaler.fit_transform(train_OHCLV_rbm)
#     train_OHCLV = scaler.fit_transform(train_OHCLV)
    train_PCA_anomalous = scaler.fit_transform(train_PCA_anomalous)
    train_PCA_autoencoder = scaler.fit_transform(train_PCA_autoencoder)
    train_PCA_rbm = scaler.fit_transform(train_PCA_rbm)
#     train_anomalous = scaler.fit_transform(train_anomalous)
#     train_autoencoder = scaler.fit_transform(train_autoencoder)
#     train_rbm = scaler.fit_transform(train_rbm)
#     train_pca = scaler.fit_transform(train_pca)

    target_returns = scaler.fit_transform(target_returns)
    # target_volatility = scaler.fit_transform(target_volatility)

    res_OHCLV_anomalous = get_predictions_dnn(train_PCA_anomalous, target_returns)
    print(' -> RMSE - RF - Returns (PCA_anomalous_RF): {}'.format(res_OHCLV_anomalous))

    res_OHCLV_autoencoder = get_predictions_dnn(train_PCA_autoencoder, target_returns)
    print(' -> RMSE - RF - Returns (PCA_autoencoder_RF): {}'.format(res_OHCLV_autoencoder))
    
#     res_OHCLV_pca = get_predictions_dnn(train_OHCLV_pca, target_returns)
#     print(' -> RMSE - RF - Returns (OHCLV_pca): {}'.format(res_OHCLV_pca))
    
    res_OHCLV_rbm = get_predictions_dnn(train_PCA_rbm, target_returns)
    print(' -> RMSE - RF - Returns (PCA_rbm_RF): {}'.format(res_OHCLV_rbm))
    
    return res_OHCLV_anomalous, res_OHCLV_autoencoder, res_OHCLV_rbm

#--- Main Module ---

assets = ['ABEV3', 'BBAS3', 'BBDC3', 'BRFS3', 'BVMF3', 'ECOR3', 'EGIE3', 'FIBR3', 'MULT3', 'PETR4', 'RAIL3', 'RENT3', 'VALE5']

friedman_matrix = []

executions = 25
for asset in assets:
        
    matrix = [0] * executions
    
    #--- Carrega e plota a série original - Fechamento (close)
    df_train = load_dataset(asset + '.csv')
    close = df_train['Close']
    #plt.title('Time Series - '+asset)
    #plt.xlabel('Number of Observations')
    #plt.ylabel('Close Value')
    #plt.plot(close)
    #plt.show()
    
    #--- Exibe a série dos returnos ---
    target_returns = (pd.read_csv('./data/t1_' + asset+ '.csv'))
    # plt.title('Time Series - Returns of ' + asset)
    # plt.xlabel('Number of Observations')
    # plt.ylabel('Return')
    # plt.plot(target_returns)
    # plt.show()
    
    # --- Exibe a série de volatilidade ---
    target_volatility = (pd.read_csv('./data/t2_' + asset + '.csv'))
    # plt.title('Time Series - Volatility of Returns ' + asset)
    # plt.xlabel('Number of Observations')
    # plt.ylabel('Volatility')
    # plt.plot(target_volatility)
    # plt.ylim((0.01, 0.02))
    # plt.show()
    
    # print('----- Teste de estacionaridade dos retornos para {}. -----'.format(asset))
    # target_returns = np.array(target_returns).reshape(-1)
    # dftest = adfuller(target_returns, maxlag=1)
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s)' % key] = value
    # print dfoutput
    
    # print('----- Teste de estacionaridade da volatilidade para {}. -----'.format(asset))
    # target_volatility = np.array(target_volatility).reshape(-1)
    # print (target_returns).shape
    # dftest = adfuller(target_volatility, maxlag=1)
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s)' % key] = value
    # print dfoutput
     
    #--- TODO: Decomposição das séries
    df = load_features_files(asset+'.csv')
    
    for i in range(executions):

        print(str(i+1) + 'º Análise e previsão de séries temporais para {}. -----'.format(asset))
        df_array = [0] * 15
        
        #--- Executa o treinamento e teste de um modelo ARIMA(0,0,1)    
        #res_arima = train_test_arima(df)
        #df_array[0] = res_arima
        
        #--- Executa o treinamento e teste de um modelo GARCH(1,1) 
        #res_garch = train_test_garch(df)
        #df_array[1] = res_garch
        
        #Executa o treinamento e teste de um ensemble com Random Forest 
        OHCLV, Anomalous, Autoencoder, PCA, RBM, OHCLV_Anomalous, OHCLV_Autoencoder, OHCLV_PCA, OHCLV_RBM, PCA_anomalous, PCA_autoencoder, PCA_rbm = train_test_dnn(df)
        df_array[0] = OHCLV
        df_array[1] = Anomalous
        df_array[2] = Autoencoder
        df_array[3] = PCA 
        df_array[4] = RBM
        df_array[5] = OHCLV_Anomalous
        df_array[6] = OHCLV_Autoencoder
        df_array[7] = OHCLV_PCA 
        df_array[8] = OHCLV_RBM
        df_array[9] = PCA_anomalous
        df_array[10] = PCA_autoencoder
        df_array[11] = PCA_rbm
        df_array[12] = PCA_anomalous
        df_array[13] = PCA_autoencoder
        df_array[14] = PCA_rbm
        

        # --- Matriz de entrada para o teste de Friedman
        #friedman_matrix.append([res_arima, res_garch,
         #                   res_OHCLV_dnn, res_anomalous_dnn, res_autoencoder_dnn, res_rbm_dnn, res_pca_dnn,
          #                  res_OHCLV_rf, res_anomalous_rf, res_autoencoder_rf, res_rbm_rf, res_pca_rf]) 
        
        #friedman_matrix.append([res_PCA_anomalous_DNN, res_PCA_autoencoder_DNN, res_PCA_rbm_DNN, res_PCA_anomalous_RF, res_PCA_autoencoder_RF, res_PCA_rbm_RF]) 
       
        matrix[i] = df_array        
        
    df = pd.DataFrame(data=matrix, columns=['OHCLV', 'Anomalous', 'Autoencoder', 'PCA, RBM', 'OHCLV_Anomalous', 'OHCLV_Autoencoder', 'OHCLV_PCA', 'OHCLV_RBM', 'PCA_anomalous', 'PCA_autoencoder', 'PCA_rbm'], dtype='float32')
    csv_name = asset + '_RF.csv'
    print ('Salvando')
    df.to_csv('Results/'+csv_name)
    print ("Salvo " + csv_name)
    
#         
# #--- Ajusta a matriz para uso da função com o teste de Friedman
# friedman_matrix = np.array(friedman_matrix).reshape(-1, 6)
# res_friedman = fu.friedman_test(friedman_matrix)
# print ('p-value - Friedman test: {}'.format(res_friedman[1]))
# 
# #--- Executa o teste post-hoc de Nemenyi
# rank = dict(zip( assets, res_friedman[2]))
# res_nemenyi = fu.nemenyi_multitest(rank)
# print (res_nemenyi[0])
# print (res_nemenyi[2])

