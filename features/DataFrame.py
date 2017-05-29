'''
Created on 16 de mai de 2017

@author: alisson
'''
import pandas as pd
import random

class Data_Frame():
    def __init__(self):
        pass
        
    def Create_DataFrame(self, num):
        nome = 'teste'
        
        arr = [0] * 5
        matrix = [0] * 5
        column = [0] * 5
        for i in range(len(matrix)):
            arr = [0] * len(column)
            for j in range(len(arr)):
                arr[j] = num
            
            matrix[i] = arr

        column[i] = {'Value 0', 'Value 1', 'Value 2', 'Value 3', 'Value 4'}

        
        
            
        df = pd.DataFrame(data=matrix)
        csv_name = 'data' + str(num) + '.csv'
        df.to_csv(csv_name)
        print ("Salvo " + csv_name)
        
        return df
            
t = Data_Frame()
data1 = t.Create_DataFrame(1)
data2 = t.Create_DataFrame(2)

data3 = pd.concat([data1, data2], axis=1)


csv_name = 'data3.csv'
data3.to_csv(csv_name)
print ("Salvo " + csv_name)

data4 = pd.read_csv('data3.csv')
data4 = data4.values
data4 = data4.astype('float32')
print(data4)
print(data4[2][0])
for i in range(len(data4)):
    if(i > 0):
        for j in range(len(data4[0])):
            print(data4[i][j], " ", end="")
        print(" ")
        
