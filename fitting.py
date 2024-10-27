# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:37:54 2024

@author: jerem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy import special
from Pulse import V_AP_P_list, V_P_AP_list, pulses

file_path = 'C:/Users/jerem/Documents/Spin-Ion Tech/Data_Example/RawData/RVdc/IV-DC_+2023_11_27_213019332_P3MRAM_WAF00_+000_+000_CH1_R07C17_AllCycle.txt'

skiprows_data = 0
with open(file_path, 'r') as file:
    for line in file:
        if '[DATA]' in line:
            break
        skiprows_data += 1

check_columns = pd.read_csv(file_path,delim_whitespace=True, header = None,skiprows = skiprows_data+1)
names_H = ['V_Pulse']
names_col = names_H+[f'Resistance_{i}' for i in range(1,len(check_columns.columns))]



test_file = pd.read_csv(file_path,delimiter = '\t', skiprows = skiprows_data+3, names = names_col)
test_file = test_file.astype(float)


#print(test_file)

#plt.plot(test_file['V_Pulse'])
plt.figure()
for i in range(2,len(check_columns.columns)):
    
    plt.plot(test_file['V_Pulse'],test_file[f'Resistance_{i}'])

print(f' THISSSSSSSSSS {V_P_AP_list}')

'''
for i in range(1,20):
    plt.plot(test_file['V_Pulse'][20*(i-1):20*i],test_file['Resistance_3'][20*(i-1):20*i])
    plt.legend()
'''

#print([i for i in range(2,len(check_columns.columns))])
test_file['V_Pulse'] /=1000

avg_file = pd.read_csv(file_path,delimiter = '\t', skiprows = skiprows_data+3, names = names_col, usecols = [i for i in range(2,len(check_columns.columns)-2)])
#average = test_file[2:].sum(1)/(len(test_file[2]))

average = avg_file.sum(1)/len(avg_file.columns)

plt.figure()

plt.plot(test_file['V_Pulse'],average)

IC_plus_list = []
IC_minus_list = []

for i in range(len(V_P_AP_list)):


    #np.nsmallest

    largest_diff = average.diff()
    
    largest_diff = largest_diff.nlargest(1).index.tolist()
    
    #print(largest_diff)
    
    
    #plt.figure()
    
    
    rc_plus = average[np.argmin(abs(test_file['V_Pulse'][:largest_diff[0]]-V_P_AP_list[i]))]
    
    print(f'this is RC+ = {rc_plus}')
    
    smallest_diff = average.diff()
    
    smallest_diff = smallest_diff.nsmallest(1).index.tolist()
    
    #print(smallest_diff)
    
    rc_minus = average[np.argmin(abs(test_file['V_Pulse'][:smallest_diff[0]]-V_AP_P_list[i]))]
    
    print(f'this is RC- = {rc_minus}')
    
    I_C_minus = 1/(rc_minus/(V_AP_P_list[i]))
    
    I_C_plus = 1/(rc_plus/(V_P_AP_list[i]))
    
    print(f'IC+ = {I_C_plus*1000}, IC- = {I_C_minus*1000}')
    
    IC_plus_list.append(I_C_plus)
    
    IC_minus_list.append(I_C_minus)
    
    #plt.plot(test_file['V_Pulse'][20:50],average[20:50])
    
    #for finding R(VC+) we want to find the value of the switching current which is still in the
    # smaller region
    
    # for finding R(VC-) we want to find the closest point on the higher point of the graph
print(IC_plus_list)
print(IC_minus_list)
plt.figure()
pulses = np.array(pulses)
pulses = 1/pulses
plt.plot(pulses,IC_minus_list)
plt.plot(pulses,IC_plus_list)

def functions(data,a,b):
    return a*data + b


popt, pcov = curve_fit(functions, pulses, IC_minus_list)

print(f'the pcov is {np.linalg.cond(pcov)}')
print(f'the pcov diag is {np.diag(pcov)}')  

print(f'these are the parameters minus = {popt}')




popt, pcov = curve_fit(functions, pulses, IC_plus_list)

print(f'the pcov is {np.linalg.cond(pcov)}')
print(f'the pcov diag is {np.diag(pcov)}')  

print(f'these are the parameters plus = {popt}')

