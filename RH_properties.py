# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:09:08 2024

@author: jerem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy import special

file_path = 'C:/Users/jerem/Documents/Spin-Ion Tech/Data_Example/RawData/RH/RH Test_+2023_11_27_212944471_P3MRAM_WAF00_+000_+000_CH1_R07C17_AllCycle.txt'


skiprows_data = 0
with open(file_path, 'r') as file:
    for line in file:
        if '[DATA]' in line:
            break
        skiprows_data += 1

check_columns = pd.read_csv(file_path,delim_whitespace=True, header = None,skiprows = skiprows_data+1)
names_H = ['Hx','Hy','Hz']
names_col = names_H+[f'Resistance_{i}' for i in range(1,len(check_columns.columns)-2)]



test_file = pd.read_csv(file_path,delimiter = '\t', header = None, skiprows = skiprows_data+3, names = names_col)
test_file = test_file.astype(float)

#print(type(test_file['Hz'][2]))

#plt.plot(test_file['Hz'],test_file['Resistance_1'])

R_AP = max(test_file['Resistance_1'])
R_P = min(test_file['Resistance_1'])


#print(f'R_AP = {R_AP} , R_P = {R_P}')

R_AP_avg = sum(test_file.max(0)[3:])/len(test_file.max(0)[3:])

R_P_avg = sum(test_file.min(0)[3:])/len(test_file.min(0)[3:])


#print(f'R_AP average = {R_AP_avg}, R_P average = {R_P_avg}')


#print(resistance_parallel)

#plt.plot(test_file['Hz'])

two_smallest = abs(test_file['Hz']).nsmallest(2)


#print(two_smallest.name)


two_smallest = two_smallest.index.tolist()


#print(two_smallest)

AP_P = test_file.iloc[two_smallest]



#print(AP_P)



#print(AP_P)


############################################## derivative option ################################


#remove all values above the largest derivative, take the average of the values before (removing)
#the values aronud the induction loop
#
#remove all values infront of the largest derivative and below the largest smallest derivative
#(below the )
R_P_avg = 0

R_AP_avg = 0

R_P = []

R_AP = []



for i in range(1,len(check_columns.columns)-2):
    largest_diff = abs(test_file[f'Resistance_{i}'].diff())
    
    
    ind_largest_diff = largest_diff.nlargest(2)
    
    ind_largest_diff = ind_largest_diff.index.tolist()
    
    #print(f'ind_largest_diff = {ind_largest_diff}')
    
    R_P_pre = test_file[f'Resistance_{i}'][:ind_largest_diff[1]]
    
    R_P_pre_2 = test_file[f'Resistance_{i}'][ind_largest_diff[0]:]
    
    #print(R_P_pre)
    
    average = (sum(R_P_pre) + sum(R_P_pre_2))/(len(R_P_pre) + len(R_P_pre_2))
    
    ind_of_real_val = np.where(test_file[f'Resistance_{i}'] < average)
    
    #print(ind_of_real_val)
    
    R_P_avg += sum(test_file[f'Resistance_{i}'].iloc[ind_of_real_val])/(len(ind_of_real_val[0]))
    
    R_P.append(sum(test_file[f'Resistance_{i}'].iloc[ind_of_real_val])/(len(ind_of_real_val[0])))
    
    ############
    
    #print(f'ind_largest_diff = {ind_largest_diff}')
    if ind_largest_diff[1]> ind_largest_diff[0]:
        R_AP_pre = test_file[f'Resistance_{i}'][ind_largest_diff[0]:ind_largest_diff[1]]
    else:
        R_AP_pre = test_file[f'Resistance_{i}'][ind_largest_diff[1]:ind_largest_diff[0]]

    average = sum(R_AP_pre)/(len(R_AP_pre))
    
    
    ind_of_real_val = np.where(test_file[f'Resistance_{i}'] > average)
    
    #print(ind_of_real_val)
    
    R_AP_avg+= sum(test_file[f'Resistance_{i}'].iloc[ind_of_real_val])/(len(ind_of_real_val[0]))
    #print(len(ind_of_real_val[0]))
    R_AP.append(sum(test_file[f'Resistance_{i}'].iloc[ind_of_real_val])/(len(ind_of_real_val[0])))
    #print(f'index R_P = {index_R_P}')
    
    #sum(R_P_pre[:index_R_P[0]])+ sum(R_P_pre[index_R_P[1]:])
    #new = test_file['Resistance_1'][:ind_largest_diff[1]].tolist()+ test_file['Resistance_1'][ind_largest_diff[0]].tolist()
    #plt.plot(new)
    
    
#print(len(check_columns.columns)-3)
    #the R_p is the average of values below this
R_P_avg = R_P_avg/(len(check_columns.columns)-3)

R_AP_avg = R_AP_avg/(len(check_columns.columns)-3)

print(f'std of R_AP = {np.std(R_AP)}')

print(f'std of R_P = {np.std(R_P)}')

print(f'R_AP_avg = {R_AP_avg} , R_P_avg = {R_P_avg}')



########################### FIND H_c ####################################

#find the middle of the switch of the resistance then find the index
#that indexes field is the coercivity

H_c_avg = 0

H_c_plus_list = []
H_c_minus_list = []

H_c_plus_loc = []

for i in range(1,len(check_columns.columns)-2):
    largest_diff = test_file[f'Resistance_{i}'].diff()
    
    ind_largest_diff = largest_diff.nlargest(1)
    
    ind_largest_diff = ind_largest_diff.index.tolist()
    
    ind_smallest_diff = largest_diff.nsmallest(1)
    
    ind_smallest_diff = ind_smallest_diff.index.tolist()
    
    #print(ind_largest_diff)
    
    #print(test_file['Resistance_1'][159])
    
    H_c_plus = test_file['Hz'][ind_largest_diff[0]]
    
    H_c_minus = test_file['Hz'][ind_smallest_diff[0]]
    
    H_c_plus_loc.append((test_file['Hz'][ind_largest_diff[0]], test_file[f'Resistance_{i}'][ind_largest_diff[0]]))
    
    H_c_plus_list.append(abs(H_c_plus))
    #print(H_c)
    
    H_c_minus_list.append(H_c_minus)
    
    #H_c_avg += abs(H_c)
    #plt.plot(test_file['Resistance_1'][650:660])
    
print(min(H_c_minus_list))    
#print(H_c_list)
avg_H_c_plus = np.mean(H_c_plus_list)
std_H_c_plus = np.std(H_c_plus_list)

avg_H_c_minus = np.mean(H_c_minus_list)
std_H_c_minus = np.std(H_c_minus_list)

print(f'std_Hc+ = {std_H_c_plus}, avg_H_c+ = {avg_H_c_plus}')

print(f'std_Hc- = {std_H_c_minus}, avg_H_c- = {avg_H_c_minus}')

#H_c_avg /= (len(check_columns.columns)-3)



#print(f'H_c average = {H_c_avg}')


TMR_avg = (1/R_P_avg - 1/R_AP_avg)/(1/R_AP_avg)*100

print(f'the average TMR = {TMR_avg}')



####################### curve fitting to get del and H_eff #############################
#R = 5Hz

def switching_probability_plus(cdf_plus,H_c,delta):
    value = (H_c)*(10**9)/(5*np.sqrt(delta))*np.sqrt(np.pi)/2*special.erfc(np.sqrt(delta)*(1-cdf_plus/H_c))
    return 1 - np.exp(-1*value)
    
    
abs_H_c_minus_list = np.abs(H_c_minus_list)


H_c_plus_norm = np.random.normal(avg_H_c_plus,std_H_c_plus)


H_c_minus_norm = np.random.normal(avg_H_c_minus,std_H_c_minus)

count_plus, bins_count_plus = np.histogram(H_c_plus_list, bins=35) 

count_minus, bins_count_minus = np.histogram(abs_H_c_minus_list, bins=35) 

pdf_plus = count_plus / sum(count_plus) 

pdf_minus = count_minus / sum(count_minus) 

cdf_plus = np.cumsum(pdf_plus) 

cdf_minus = np.cumsum(pdf_minus)

cdf_plus = np.cumsum(pdf_plus) 

print(count_plus.size)

#switching_probability_plus = 1-np.exp((H_c)*(10**9)/(5*np.sqrt(delta))*np.sqrt(np.pi)/2*special.erfc(np.sqrt(delta)*(1-cdf_plus/H_c)))

#plt.plot(bins_count_plus[1:], pdf_plus, color="red", label="PDF+") 

#plt.plot(bins_count_minus[1:], pdf_minus, color="orange", label="PDF-") 

#plt.plot(bins_count_plus[1:], cdf_plus, label="CDF+") 

#plt.plot(bins_count_minus[1:], cdf_minus, label="CDF-") 
#plt.legend() 

#plt.plot(H_c_plus_list, 'bo')

#print(cdf_plus)
#print(bins_count_plus[1:])
x = np.linspace(.1,.5,100)

popt, pcov = curve_fit(switching_probability_plus, bins_count_plus[1:], cdf_plus,p0 = [.34,68], bounds = ([10**(-9),10],[np.inf,np.inf]))

#for i in x:
#    for j in range(1,100):
        #try:
#            popt, pcov = curve_fit(switching_probability_plus, bins_count_plus[1:], cdf_plus, p0 = [i,j], bounds = ([0.01,0.01],[100,1000]))
        #except RuntimeWarning:
        #    continue
        #except ValueError:
        #    continue
        
print(f'the pcov is {np.linalg.cond(pcov)}')
print(f'the pcov diag is {np.diag(pcov)}')  
plt.figure()
plt.subplot(2,1,1)

print(f'popt = {popt}')

print(f'pcov is {pcov}')

plt.plot(bins_count_plus[1:], switching_probability_plus(bins_count_plus[1:], *popt), 'g--')


plt.subplot(2,1,2)

plt.plot(bins_count_plus[1:],cdf_plus)

R_P = np.array(R_P)
R_AP = np.array(R_AP)


TMR = (1/R_P - 1/R_AP)/(1/R_AP)*100


plt.figure()

plt.subplot(2,3,1)
plt.plot(H_c_minus_list,'bo')

plt.subplot(2,3,2)
plt.plot(H_c_plus_list, 'bo')

plt.subplot(2,3,3)


plt.subplot(2,3,4)
plt.plot(R_P,'bo')

plt.subplot(2,3,5)
plt.plot(R_AP,'bo')

plt.subplot(2,3,6)

plt.plot(TMR,'bo')



print(cdf_plus)


plt.figure()
delta = 68
H = .34


plt.plot(bins_count_plus[1:],(1-np.exp(-1*(H)*(10**9)/(5*np.sqrt(delta))*np.sqrt(np.pi)/2*special.erfc(np.sqrt(delta)*(1-cdf_plus/H)))))





