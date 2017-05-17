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
    def __init__(self, train_OHLCV, train_anomalous, train_autoencoder, train_rbm, train_pca, target_returns, target_volatility):
        self._train_OHLCV = train_OHLCV
        self._train_anomalous = train_anomalous
        self._train_autoencoder = train_autoencoder
        self._train_rbm = train_rbm
        self._train_pca = train_pca
        self._target_returns = target_returns
        self._target_volatility = target_volatility

    def __train_OHLCV(self):
        return self._train_OHLCV

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



def load_dataset(asset):

    #--- Carrega a série original    
    #home_dir = os.getenv("HOME")
    print (' -> Loading asset file: %s' % asset),
    df_train = pd.read_csv('./data/' + asset)
    print ' -> Done.'
    print (' -> Original dataset dimensions: {} samples, with {} attributes.'.format(df_train.shape[0], df_train.shape[1]))
    return df_train

def load_features_files(asset):

    train_OHLCV = np.loadtxt('./data/001_' + asset, delimiter=',') # Open, High, Low, Close, Volume
    train_anomalous = np.loadtxt('./data/002_' + asset, delimiter=',') # Features extraídas do pacote R de Rob Hyndman
    train_autoencoder = np.loadtxt('./data/003_' + asset, delimiter=',') # Features extraídas através de um autoencoder
    train_rbm = np.loadtxt('./data/004_' + asset, delimiter=',') # Features extraídas com uma RBM
    train_pca = np.loadtxt('./data/005_' + asset, delimiter=',') # Features extraídas com PCA

    target_returns = np.loadtxt('./data/t1_' + asset, delimiter=',') # Target 1: Retorno do próximo dia
    target_volatility = np.loadtxt('./data/t2_' + asset, delimiter=',') # Target 2: Volatilidade do próximo dia

    df = Features(train_OHLCV=train_OHLCV, train_anomalous=train_anomalous, train_autoencoder=train_autoencoder,
                  train_rbm=train_rbm, train_pca=train_pca, target_returns=target_returns, target_volatility=target_volatility)

    return df

warnings.filterwarnings("ignore")
def train_test_arima(df):

    target_returns = df._target_returns
    # target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()
    target_returns = scaler.fit_transform(target_returns)
    # target_volatility = scaler.fit_transform(target_volatility)

    rmse_returns_ohlcv = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target_returns):
        y_train_OHLCV, y_test_OHLCV = np.transpose(target_returns)[train_index], np.transpose(target_returns)[test_index]

        # order = fu.get_PDQ_parallel(y_train_OHLCV, multiprocessing.cpu_count())
        # print ('Estimated order for ARIMA model: {}'.format(order))
        order = (0,0,1)

        model_returns_ohlcv = ARIMA(y_test_OHLCV, order=order)
        results_AR = model_returns_ohlcv.fit(disp=-1)
        pred = results_AR.fittedvalues

        rmse = np.sqrt(mean_squared_error(y_test_OHLCV, pred))
        # print('RMSE - OHLCV: {}'.format(rmse))
        rmse_returns_ohlcv.append(rmse)

    res = np.mean(rmse_returns_ohlcv)
    print(' -> RMSE - ARIMA - Returns: {}'.format(res))

    return res

warnings.filterwarnings("ignore")
def train_test_garch(df):

    target_returns = df._target_returns
    scaler = preprocessing.MinMaxScaler()
    target_returns = scaler.fit_transform(target_returns)
    rmse_returns_ohlcv = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target_returns):
        y_train_OHLCV, y_test_OHLCV = np.transpose(target_returns)[train_index], np.transpose(target_returns)[
            test_index]

        model_returns_ohlcv = arch_model(y_test_OHLCV, vol='Garch', p=1, o=0, q=1, dist='Normal')
        results_GARCH = model_returns_ohlcv.fit(update_freq=10, disp='off')
        pred = results_GARCH.resid

        rmse = np.sqrt(np.mean(np.power(pred,2)))
        rmse_returns_ohlcv.append(rmse)

    res = np.mean(rmse_returns_ohlcv)
    print(' -> RMSE - GARCH(1,1) - Returns: {}'.format(res))

    return res

def get_predictions_dnn(train, target):

    rmse_returns = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target):
        # --- Cross validation - OHLCV

        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = np.transpose(target)[train_index], np.transpose(target)[
            test_index]

        y_train = np.transpose(y_train)
        y_test = np.transpose(y_test)

        model = Sequential()
        model.add(Dense(64, input_dim=(X_train.shape)[1], init='normal', activation='relu'))
        model.add(Dense(32, init='normal', activation='relu'))
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

    train_OHLCV = df._train_OHLCV
    train_anomalous = df._train_anomalous
    train_autoencoder = df._train_autoencoder
    train_rbm = df._train_rbm
    train_pca = df._train_pca

    target_returns = df._target_returns
    target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()

    train_OHLCV = scaler.fit_transform(train_OHLCV)
    train_anomalous = scaler.fit_transform(train_anomalous)
    train_autoencoder = scaler.fit_transform(train_autoencoder)
    train_rbm = scaler.fit_transform(train_rbm)
    train_pca = scaler.fit_transform(train_pca)

    target_returns = scaler.fit_transform(target_returns)
    # target_volatility = scaler.fit_transform(target_volatility)

    res_OHLCV = get_predictions_dnn(train_OHLCV, target_returns)
    print(' -> RMSE - DNN - Returns (OHLCV): {}'.format(res_OHLCV))

    res_anomalous = get_predictions_dnn(train_anomalous, target_returns)
    print(' -> RMSE - DNN - Returns (Anomalous): {}'.format(res_anomalous))

    res_autoencoder = get_predictions_dnn(train_autoencoder, target_returns)
    print(' -> RMSE - DNN - Returns (Autoencoder): {}'.format(res_autoencoder))

    res_rbm = get_predictions_dnn(train_rbm, target_returns)
    print(' -> RMSE - DNN - Returns (RBM): {}'.format(res_rbm))

    res_pca = get_predictions_dnn(train_pca, target_returns)
    print(' -> RMSE - DNN - Returns (PCA): {}'.format(res_pca))

    return res_OHLCV, res_anomalous, res_autoencoder, res_rbm, res_pca

def get_predictions_rf(train, target):

    rmse_returns = []

    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(target):
        # --- Cross validation - OHLCV

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

    train_OHLCV = df._train_OHLCV
    train_anomalous = df._train_anomalous
    train_autoencoder = df._train_autoencoder
    train_rbm = df._train_rbm
    train_pca = df._train_pca

    target_returns = df._target_returns
    target_volatility = df._target_volatility

    scaler = preprocessing.MinMaxScaler()

    train_OHLCV = scaler.fit_transform(train_OHLCV)
    train_anomalous = scaler.fit_transform(train_anomalous)
    train_autoencoder = scaler.fit_transform(train_autoencoder)
    train_rbm = scaler.fit_transform(train_rbm)
    train_pca = scaler.fit_transform(train_pca)

    target_returns = scaler.fit_transform(target_returns)
    # target_volatility = scaler.fit_transform(target_volatility)

    res_OHLCV = get_predictions_rf(train_OHLCV, target_returns)
    print(' -> RMSE - RF - Returns (OHLCV): {}'.format(res_OHLCV))

    res_anomalous = get_predictions_rf(train_anomalous, target_returns)
    print(' -> RMSE - RF - Returns (Anomalous): {}'.format(res_anomalous))

    res_autoencoder = get_predictions_rf(train_autoencoder, target_returns)
    print(' -> RMSE - RF - Returns (Autoencoder): {}'.format(res_autoencoder))

    res_rbm = get_predictions_rf(train_rbm, target_returns)
    print(' -> RMSE - RF - Returns (RBM): {}'.format(res_rbm))

    res_pca = get_predictions_rf(train_pca, target_returns)
    print(' -> RMSE - RF - Returns (PCA): {}'.format(res_pca))

    return res_OHLCV, res_anomalous, res_autoencoder, res_rbm, res_pca

#--- Main Module ---

assets = ['ABEV3', 'BBAS3', 'BBDC3', 'BRFS3', 'BVMF3', 'ECOR3', 'EGIE3', 'FIBR3', 'MULT3', 'PETR4', 'RAIL3', 'RENT3', 'VALE5']

friedman_matrix = []

executions = 25
for asset in assets:
    
    
    matrix = [0] * executions
    
    for i in range(executions):

        df_array = [0] * len(assets)
        print(str(i+1) + 'º Análise e previsão de séries temporais para {}. -----'.format(asset))

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
        
        #--- Executa o treinamento e teste de um modelo ARIMA(0,0,1)    
        res_arima = train_test_arima(df)
        df_array[0] = res_arima
        
        #--- Executa o treinamento e teste de um modelo GARCH(1,1) 
        res_garch = train_test_garch(df)
        df_array[1] = res_garch

        #--- Executa o treinamento e teste de uma rede neural com 3 camadas
        res_OHLCV_dnn, res_anomalous_dnn, res_autoencoder_dnn, res_rbm_dnn, res_pca_dnn = train_test_dnn(df)
        df_array[2] = res_OHLCV_dnn
        df_array[3] = res_anomalous_dnn 
        df_array[4] = res_autoencoder_dnn 
        df_array[5] = res_rbm_dnn 
        df_array[6] = res_pca_dnn
        
        #--- Executa o treinamento e teste de um ensemble com Random Forest 
        res_OHLCV_rf, res_anomalous_rf, res_autoencoder_rf, res_rbm_rf, res_pca_rf = train_test_rf(df)
        df_array[7] = res_OHLCV_rf 
        df_array[8] = res_anomalous_rf 
        df_array[9] = res_autoencoder_rf 
        df_array[10] = res_rbm_rf 
        df_array[11] = res_pca_rf

        # --- Matriz de entrada para o teste de Friedman
        friedman_matrix.append([res_arima, res_garch,
                            res_OHLCV_dnn, res_anomalous_dnn, res_autoencoder_dnn, res_rbm_dnn, res_pca_dnn,
                            res_OHLCV_rf, res_anomalous_rf, res_autoencoder_rf, res_rbm_rf, res_pca_rf]) 
       
        matrix[i] = df_array
        
        
    print matrix
    df = pd.DataFrame(data=matrix, columns=['Arima', 'GARCH'], dtype='float32')
    csv_name = asset + '.csv'
    print 'Salvando'
    df.to_csv('Results/'+csv_name)
    print ("Salvo " + csv_name)
    
        
#--- Ajusta a matriz para uso da função com o teste de Friedman
friedman_matrix = np.array(friedman_matrix).reshape(-1, 13)
res_friedman = fu.friedman_test(friedman_matrix)
print ('p-value - Friedman test: {}'.format(res_friedman[1]))

#--- Executa o teste post-hoc de Nemenyi
rank = dict(zip( assets, res_friedman[2]))
res_nemenyi = fu.nemenyi_multitest(rank)
print res_nemenyi[0]
print res_nemenyi[2]





