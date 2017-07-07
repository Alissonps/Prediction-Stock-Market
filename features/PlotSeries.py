'''
Created on 19 de jun de 2017

@author: alisson
'''
import numpy as np

a = [0,1,3,5,7]
b = [0,2,4,6,8]

c = []
c = np.concatenate([c, b], axis=0)

print(c)

c = np.concatenate([c, a], axis=0)

print(c)

c = np.concatenate([c, a], axis=0)

print(c)
