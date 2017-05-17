'''
Created on 5 de mai de 2017

@author: alisson
'''
import random
import math
import pandas
import numpy as np
import copy
import matplotlib.pyplot as plt
from theano.compile.io import Out
from wsgiref.validate import ErrorWrapper

class PSO():
    def __init__(self, particles_number, particles_dimension, len_window, input, target, iterations):
        self.particles = [0.0] * particles_number
        self.fitness = [0.0] * particles_number
        self.g_best = [0.0] * particles_dimension
        self.p_best = [0.0] * particles_number
        self.fit_pBest = [0.0] * particles_number
        self.g_fitness = 0
        self.velocity = [0.0] * particles_number
        self.inercia = 0.5
        self.social_factor = 2.0
        self.individual_factor = 2.0
        self.particles_number = particles_number
        self.particles_dimension = particles_dimension
        self.len_window = len_window
        self.input = input
        self.target = target
        self.iterations_number = iterations
        
    def Begin_Particles(self):
        
        particle = [0.0] * self.particles_dimension
        
        for i in range(self.particles_number):
            for j in range(self.particles_dimension):
                dimension = random.random()
                particle[j] = dimension
            
            self.particles[i] = particle
            self.p_best[i] = particle
        
        self.Fitness()
        self.g_fitness = self.fitness[0]
        self.fit_pBest = self.fitness
        self.Best()
        
        velocity = [0.0] * self.particles_dimension
        for i in range(len(self.velocity)):          
            self.velocity[i] = velocity
        
        self.Velocity()
        
    def Fitness(self):
        
        yolcu = Yolcu(self.input, self.target)
        predictions = [0.0] * self.particles
        
        for i in range(len(self.particles)):
            lin_input_w = self.particles[i][0:self.len_window]
            nolin_input_w = self.particles[i][len(lin_input_w):(len(lin_input_w)+self.len_window)]
            out_w = self.particles[i][(len(nolin_input_w)+len(lin_input_w)):(len(nolin_input_w)+len(lin_input_w))+2]
            input_bias = self.particles[i][(self.particles_dimension-3):(self.particles_dimension-1)]
            out_bias = self.particles[i][(self.particles_dimension-1):]
            
            yolcu.Set_Weights(lin_input_w, nolin_input_w, out_w, input_bias, out_bias)
            
            prediction = [0.0] * len(self.target)
            error = [0.0] * len(self.target)
            
            for j in range(len(self.input)):
                prediction[j] = yolcu.Neural_Net(self.input[j])
                error[j] = (self.target[j] - prediction[j])
            
            predictions[i] = prediction
            self.fitness[i] = yolcu.MSE(error)
        return predictions
    
    def Best(self):
        print "Bests"
        
        for i in range(len(self.particles)):
            if(self.fitness[i] < self.g_fitness):
                self.g_fitness = self.fitness[i]
                self.g_best = self.particles[i]
            if(self.fitness[i] < self.fit_pBest[i]):
                self.fit_pBest[i] = self.fitness[i]
                self.p_best[i] = self.particles[i]

    def Velocity(self):
        print "Velocity"
        
        for i in range(len(self.particles)):
            for j in range(len(self.particles[0])):
                rand1 = random.random()
                rand2 = random.random()
                social = self.g_best[j] * self.social_factor * rand1
                individual = self.p_best[i][j] * self.individual_factor *  rand2
                self.velocity[i][j] = self.velocity[i][j]  + social + individual
    
        pass

    def Particle_update(self):
        print "Particle Update"
        
        print self.particles[1]
        for i in range(len(self.particles)):
            for j in range(len(self.particles[0])):
                self.particles[i][j] = self.particles[i][j] + self.velocity[i][j]
                
                if(self.particles[i][j] > 1):
                    self.particles[i][j] = 1
                elif(self.particles[i][j] < -1):
                    self.particles[i][j] = -1
                
        print self.particles[1]

    def Train(self):
        
        print "Train"
        
        self.Begin_Particles()
        
        error = [0.0] * self.iterations_number
        for i in range(self.iterations_number):
            self.Fitness()
            self.Best()
            self.Velocity()
            self.Particle_update()
        
            print self.g_fitness    
            error[i] = self.g_fitness 
            
        
        
        return error
    
class Yolcu():
    
    def __init__(self, input = None, target = None):
        
        self.input = input
        self.target = target
        
        self.linear_neuron = 0.0
        self.noLinear_neuron = 0.0
        self.output_neuron = 0.0
        self.output_bias = 1.0
        self.sum_linear = 0.0
        self.sum_noLinear = 0.0
        self.output = 0.0
        self.change_linear_output = 1.0
        self.change_noLinear_output = 1.0
        self.change_linear_input = 1.0
        self.change_noLinear_input = 1.0
        self.output_noLinear = 1
        
        self.linear_input_weights = []
        self.noLinear_input_weights = []
        self.output_weights = []
        self.error = []
        
        self.input_bias = [1,1]
        self.output_weights = [random.random(), random.random()]
        
#     def Weights_start(self):
#         for i in range(len(self.input)):
#             self.linear_input_weights.append(random.random())
#             self.noLinear_input_weights.append(random.random())
#             
    
    def Set_Weights(self, input_lin, input_noLin, out, input_bias, out_bias):
        self.linear_input_weights = input_lin
        self.noLinear_input_weights = input_noLin
        self.output = out
        self.input_bias = input_bias
        self.output_bias = out_bias
        
    def Neural_Net(self, input):
        
        for i in range(len(input)):
            self.sum_linear = self.sum_linear + (input[i] * self.linear_input_weights[i])
            self.sum_noLinear = self.sum_noLinear + (input[i] * self.noLinear_input_weights[i])
            
        linear_function = self.sum_linear + self.input_bias[0]
        self.sum_noLinear = self.sum_noLinear + self.input_bias[1]
        
        noLinear_function = self.Sigmoide(self.sum_noLinear)
        
        self.linear_neuron = linear_function
        self.noLinear_neuron = noLinear_function
        
        self.output_linear = self.linear_neuron * self.output_weights[0]
        self.output_noLinear = self.noLinear_neuron * self.output_weights[1]
        
        self.sum_output = self.output_linear + self.sum_noLinear + self.output_bias
        
        self.output_neuron = self.Sigmoide(self.sum_output)

        self.output = self.output_neuron
        
        return self.output
    
    def Sigmoide(self, x):
        
        x = abs(x)
        sig = 1 / (1+math.exp(-x))
        
        return sig
      
    def MSE(self, error):
        
        sum = 0
        for i in range(len(error)):
            sum = sum + (error[i] * error[i])
            
        sum = sum / len(error)
        
        return sum

def Normalizar(serie):

    min = copy.deepcopy(np.min(serie))
    max = copy.deepcopy(np.max(serie))
        
    serie_norm = []
        
    for e in serie:
        valor = (e - min)/(max - min)
        serie_norm.append(valor)
    
    return serie_norm
      
def Data(data, len_window):
        
    dataset = []
    targ = []
    
    data_norm = Normalizar(data)
        
    for i in range(len(data_norm)-(len_window+1)):
        dataset.append(data_norm[i:i+len_window])
        targ.append(data_norm[i+len_window+1])

    return dataset, targ    
 
        
data = pandas.read_csv('teste.csv', usecols=[0],engine='python')
datas = data.values
len_window = 5

[input, target] = Data(datas, len_window)
    
#input = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.5, 0.6, 0.7], [0.3, 0.1, 0.5, 0.8], [0.4, 0.9, 0.1, 0.2], [0.5, 0.5, 0.7, 0.8]]
#target = [0.3, 1.04, 0.99, 1.02, 1.63]
dimension = (len_window * 2) + 2
    
teste = PSO(10, dimension, len_window, input, target, 5)

teste.Begin_Particles()
error = teste.Train()
    
#plt.plot(target, label = 'Real', color = 'Blue')
#plt.plot(pred, label = 'Previsao', color = 'Red')
plt.plot(error, label = 'Previsao', color = 'Red')
#print(pred)
#print(target)
print(error)
plt.legend()
plt.tight_layout()
plt.show()
