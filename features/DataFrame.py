'''
Created on 16 de mai de 2017

@author: alisson
'''
import pandas as pd
import random

class Data_Frame():
    def __init__(self):
        pass
        
    def Create_DataFrame(self):
        nome = 'teste'
        
        arr = [0] * 5
        matrix = [0] * 5
        column = [0] * 5
        for i in range(len(matrix)):
            arr = [0] * len(column)
            for j in range(len(arr)):
                arr[j] = random.random()
            
            matrix[i] = arr

        column[i] = {'Value 0', 'Value 1', 'Value 2', 'Value 3', 'Value 4'}

        
        
        
        df = pd.DataFrame(data=matrix,columns=column)
    
        csv_name = nome + '.csv'
        print 'Salvando'
        df.to_csv(csv_name)
        print ("Salvo " + csv_name)
            
t = Data_Frame()
t.Create_DataFrame()