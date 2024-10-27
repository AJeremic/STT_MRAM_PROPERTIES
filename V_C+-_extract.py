# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:56:18 2024

@author: jerem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pulses = [10,15,20,30,50,75]


this = ['/insert_data_here']


skiprows_data = 0

with open(this[0], 'r') as file:
    for line in file:
        if '[DATA]' in line:
            break
        skiprows_data += 1

check_columns = pd.read_csv(this[0],delim_whitespace=True, header = None,skiprows = skiprows_data+1)
names_H = ['Pulse_A','Pulse_W']
names_col = names_H+[f'Resistance_{i}' for i in range(1,len(check_columns.columns)-1)]


test_file_1 = pd.read_csv(this[0],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_1 = test_file_1.astype(float)

test_file_2 = pd.read_csv(this[1],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_2 = test_file_2.astype(float)

test_file_3 = pd.read_csv(this[2],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_3 = test_file_3.astype(float)

test_file_4 = pd.read_csv(this[3],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_4 = test_file_4.astype(float)

test_file_5 = pd.read_csv(this[4],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_5 = test_file_5.astype(float)

test_file_6 = pd.read_csv(this[5],delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file_6 = test_file_6.astype(float)



print(test_file_1['Resistance_1'])

#print(f'value of Resistance_1 at .288 = {test_file["Resistance_1"]}')

#print(f'value of Resistance_1 at before .288 = {test_file["Resistance_1"]}')

#print(test_file['Resistance_1'])

#print(f'max of the V values are {max(test_file["Pulse_A"])}')
#for i in range(1,len(check_columns.columns)-2):
    
#    plt.plot(test_file['Pulse_A'],test_file[f'Resistance_{i}'])

#print(test_file['Resistance_1'].size)

    
V_AP_P_list = []
    
V_P_AP_list = []

for j in this:
    test_file_1 = pd.read_csv(j,delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
    test_file_1 = test_file_1.astype(float)
    
    list_VC_plus = []
    
    list_VC_minus = []
    
    
    for i in range(1,len(check_columns.columns)-2):
        
        largest_diff = test_file_1[f'Resistance_{i}'].diff()
        
        ind_largest_diff = largest_diff.nlargest(1)
        
        ind_largest_diff = ind_largest_diff.index.tolist()
        
        list_VC_plus.append(test_file_1['Pulse_A'][ind_largest_diff[0]])
        
        #print(ind_largest_diff[0])
        
        smallest_diff = test_file_1[f'Resistance_{i}'].diff()
        
        ind_smallest_diff = smallest_diff.nsmallest(1)
        
        ind_smallest_diff = ind_smallest_diff.index.tolist()
        
        list_VC_minus.append(test_file_1['Pulse_A'][ind_smallest_diff[0]])
        
    print(f'V_P_AP = {np.mean(list_VC_plus)}')
    
    print(f'V_AP_P = {np.mean(list_VC_minus)}')
        

    
    V_AP_P = np.mean(list_VC_minus)
    
    V_AP_P_list.append(np.mean(list_VC_minus))
    
    V_P_AP = np.mean(list_VC_plus)
    
    V_P_AP_list.append(np.mean(list_VC_plus))
    
    #def switching_positive()

    
        
        


