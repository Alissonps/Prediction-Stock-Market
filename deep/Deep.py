#-*- coding: utf-8 -*-
'''
Created on 15 de mai de 2017

@author: alisson
'''



import pandas
import math
import numpy
import keras
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Dropout
from datetime import datetime
import matplotlib.pyplot as plt

from keras.layers.normalization import BatchNormalization
from Tkinter import Label
#import matplotlib.pyplot as plt

class Deep():
    def set_data(self, data, look_back):
        
        # fix random seed for reproducibility
        #numpy.random.seed(7)
        # load the dataset
        dataframe = pandas.read_csv(data, usecols=[0], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        # normalize the dataset
        print dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        # reshape into X=t and Y=t+1
        
        trainX, trainY = self.create_dataset(train, look_back)
        testX, testY = self.create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1]))
        
        return trainX, trainY, testX, testY, scaler, dataset
        
    def Deep_network(self, trainX, trainY, testX, testY, scaler, look_back, b_size, epocas, dataset):
        self.look_back = look_back
        self.trainX =trainX
        self.trainY =trainY
        self.testX = testX 
        self.testY = testY
        self.b_size = b_size
        self.epocas = epocas
        self.dataset = dataset
        
        print("create and fit the LSTM network")
        model = Sequential()
        
        
        '''Para experimentos com apenas 1 camda LSTM usar a linha abaixo e comentar as demais linhas de LSTM'''
        
        model.add(Dense(look_back, input_dim=look_back, activation="sigmoid"))
        model.add(Dense(look_back, input_dim=look_back, activation="sigmoid"))
        model.add(Dense(look_back, input_dim=look_back, activation="sigmoid"))
        model.add(Dense(look_back, input_dim=look_back, activation="sigmoid"))
        model.add(Dense(1, input_dim=look_back, activation="sigmoid"))
        
        print("Usando o método Compile")
        model.compile(loss='mean_squared_error', optimizer='adam')
        print("Usando o método Fit")
        
        '''O parametro 'verbose' altera a exibição do treinamento. Para exibir da forma como já vinhamos fazendo use 'verbose=2'.
        'verbose=0' não exibe nada durante o treinamento'''
        '''print("Inicial time: " + str(datetime.now().strftime("%H %M %S")))'''
        model.fit(self.trainX, self.trainY, nb_epoch=self.epocas, batch_size=self.b_size, verbose=0)
        '''print("Final time: " + str(datetime.now().strftime("%H %M %S")))'''
        
        print("Fazendo as predições")
        trainPredict = model.predict(self.trainX, batch_size=self.b_size)
        testPredict = model.predict(self.testX, batch_size=self.b_size)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        print("\n")
       
        
        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(self.dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, ] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(self.dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(self.dataset)-1, ] = testPredict
        # plot baseline and predictions
        print self.dataset
        print scaler.inverse_transform(self.dataset)
        print trainPredict
        plt.plot(scaler.inverse_transform(self.dataset), Label="Serie Real")
        plt.plot(trainPredictPlot, Label="Treinamento")
        plt.plot(testPredictPlot, Label="Teste")
        plt.legend(loc='upper rigth')
        plt.tight_layout()
        plt.show()
        
        
        return trainScore, testScore
        
    def create_dataset(self, dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    
    
    
data = 'teste.csv'
t = Deep()
look_back = 12
[trainX, trainY, testX, testY, scaler, dataset] = t.set_data(data, look_back)
t.Deep_network(trainX, trainY, testX, testY, scaler, look_back, 5, 1000, dataset)



