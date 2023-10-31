# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:58:43 2022

@author: suyan
"""
import gurobipy as gp
from gurobipy import GRB
import  numpy as np
import pandas as pd
import pickle
import random
import time

# Record the start time
start_time = time.time()


num_health_districts = 26
time_periods = 6
od_flow = pd.read_csv('data/mean_df_20210224_20210424.csv')
#c_m = pd.read_csv('C_matrix.csv')
#c_m_v = c_m.values[:,1:]
with open("data/travel_time.pkl", "rb") as file:
    c_m_v = pickle.load(file)
#c_matrix is the inconvinence cost matrix
c_matrix =np.array([np.array([float(i) for i in j ])for j in c_m_v])
# =============================================================================
# c_matrix = np.eye(26)
# for i in range(26):
#     for j in range(26):
#         if j<i:
#             c_matrix[i,j] = abs(j-i)
#         else:
#              c_matrix[i,j] = abs(j-i)*1.1   
# =============================================================================

#value is the od matrix
value = od_flow.values[0:-1,1:-1]
value =np.array([np.array([float(i) for i in j ])for j in value])
actualcommunter_scaler = 4753898/np.sum(value)
value = value*actualcommunter_scaler
#################
#Symmetric
#################
# =============================================================================
# for i in range(len(value)):
#     for j in range(len(value)):
#         if i>j:
#             value[i,j]=value[j,i]
# =============================================================================
################
flowin = np.sum(value,axis=0)
flowout = np.sum(value,axis =1)
#pop over 60 by health districts
#adjust od so that total flow < population
totalpop = [420697,443569,344450,526877,899111,339399,419797,308499,140361,547523,354750,479505,287613,666399,278815,193899,166374,379199,356465,195082,407864,321720,201739,411617,469439,465691]
districts = {0: 'Antelope Valley', 
             1: 'East Valley', 
             2: 'Glendale', 
             3: 'San Fernando', 
             4: 'West Valley',
             5: 'Alhambra', 
             6: 'El Monte', 
             7: 'Foothill', 
             8: 'Pasadena', 
             9: 'Pomona',
             10: 'Central', 
             11: 'Hollywood-Wilshire', 
             12: 'Northeast', 
             13: 'West',
             14: 'Compton', 
             15: 'South', 
             16: 'Southeast', 
             17: 'Southwest',
             18: 'Bellflower', 
             19: 'East Los Angeles', 
             20: 'San Antonio', 
             21: 'Whittier',
             22: 'Harbor', 
             23: 'Inglewood', 
             24: 'Long Beach', 
             25: 'Torrance'}
for i in range(26):
    if totalpop[i]<max(flowin[i],flowout[i]):
        for j in range(26):
            value[i,j] = value[i,j]*(totalpop[i]/flowout[i])
            value[j,i] = value[j,i]*(totalpop[i]/flowin[i])
flowin = np.sum(value,axis=0)
flowout = np.sum(value,axis =1)
for i in range(26):
    for j in range(26):
        value[i,j] = int(value[i,j])
value_weights = value.copy()
#value = np.zeros((26,26))

emp=[0.9785489423063246,
0.9749523393023726,
0.9810134958440276,
0.9715990029226316,
0.9512510687291531,
0.9793683522808072,
0.9739319327227332,
0.9817156078851325,
0.9915252477779021,
0.9678482424468472,
0.9760508448195166,
0.9657752125951641,
0.9784091897580796,
0.967984322913777,
0.9795198849727248,
0.9859672856449567,
0.9874863827092342,
0.9723956829474802,
0.9745472149882571,
0.985468457303294,
0.9697058020586842,
0.9771345654416476,
0.9898114604005365,
0.979337971566636,
0.9766137484373549,
0.9772144560727959]
################################
#compute dijk
################################
D = np.zeros((26,26,26))
ccc= 0
for i in range(26):
    for j in range(26):
        for k in range(26):
            D[i,j,k] = min(c_matrix[i][k]+c_matrix[k][i],c_matrix[j][k]+c_matrix[k][j],c_matrix[j][k]+c_matrix[k][i]-c_matrix[j][i],c_matrix[i][k]+c_matrix[k][j]-c_matrix[i][j])
            if D[i,j,k]<0:
                D[i,j,k]=0
# =============================================================================
# lista = [0,6,10,17,24]
# D = np.ones((26,26,26))*10
# for i in range(26):
#     for j in range(26):
#         D[i,j,i] = 0
#         for k in range(26):
#             if i not in lista:
#                 D[i,j,k]=0
# c_matrix = np.ones((26,26))*10
# for i in range(26):
#     c_matrix[i,i]=0
#     for j in range(26):
#         if i not in lista:
#             c_matrix[i,j]=0
# =============================================================================
#formulation including inconvinence
sol = []
#limit_site = number of sites
# =============================================================================
# array([0.21334436, 0.29305546, 0.03615045, 0.07924728, 0.46735506,
#        0.29401151, 0.53411346, 0.18048311, 0.67160676, 0.40473816,
#        0.39133262, 0.82464918, 0.00766008, 0.90730261, 0.43333706,
#        0.67305033, 0.16789291, 0.24846809, 0.87774574, 0.64764514,
#        0.64652597, 0.46743305, 0.35671903, 0.66164728, 0.00978716,
#        0.35323258])
# =============================================================================
#HL = np.ones(26)*np.array([random.random() for _ in range(26)])*totalpop
HL = np.ones(26)*totalpop*emp
with open("Results/base_od_4time_emp_10lambda_popweighted.pkl", "rb") as file:
    initials = pickle.load(file)
dfxrand = initials[initials.name=='x']
dfyrand = initials[initials.name=='y']
dfzrand = initials[initials.name=='z']
groupedx = dfxrand.groupby(['i'])
resultsx = groupedx.sum()
groupedy = dfyrand.groupby(['i', 'j','t'])
resultsy = groupedy.sum()
groupedz = dfzrand.groupby(['i', 'j','k','t'])
resultyz = groupedz.sum()
for limit_site in range(6,7):
    print('#######################################################################')
    with open('data/weights_bc.pkl', 'rb') as file:
        weights_bc = pickle.load(file)
    weights_bc = totalpop.copy()
    weights_bc = 50*np.array(np.sum(value_weights,axis=1))/np.sum(np.array(value_weights))
    weights = 50*np.array(weights_bc)/np.sum(np.array(weights_bc))
    lambda_ = 10
    try:
    
        # Create a new model
        lm = gp.Model("lm")
    
        # Create variables
        
        x = lm.addVars(num_health_districts,vtype=GRB.BINARY, name="x") 
        y = lm.addVars(num_health_districts,num_health_districts,time_periods,vtype=GRB.INTEGER, name="y")
        z = lm.addVars(num_health_districts,num_health_districts,num_health_districts,time_periods,vtype=GRB.INTEGER, name="z")
        vv = lm.addVars(time_periods,num_health_districts,vtype=GRB.CONTINUOUS, name="v") 
        s = lm.addVars(2,vtype=GRB.CONTINUOUS, name="s") 
        ##initial
        for i in range(num_health_districts):
            x[i].start = resultsx['value'][i]
            #lm.addConstr(x[i] == resultsx['value'][i])
        for t in range(time_periods):
            for i in range(num_health_districts):
                for j in range(num_health_districts):
                    y[i,j,t].start = resultsy['value'][(i,j,t)]
                    for k in range(num_health_districts):
                        z[i,j,k,t].start = resultyz['value'][(i,j,k,t)]
        # Set objective
        ##
        cost_y = gp.quicksum(y[i, j, t] * (2 * c_matrix[i][j]) for i in range(num_health_districts)
                     for j in range(num_health_districts) for t in range(time_periods))
        cost_z = gp.quicksum(z[i, j, k, t] * D[i][j][k] for i in range(num_health_districts)
                     for j in range(num_health_districts) for k in range(num_health_districts)
                     for t in range(time_periods))
        
        cost_herd = gp.quicksum(vv[t,i]*weights[i]*(0.9**t) for t in range(time_periods) for i in range(num_health_districts))
        #lm.setObjective(cost_y+cost_z+lambda_*cost_herd+lambda_*15*(s[0]+s[1]),GRB.MINIMIZE)
        lm.setObjective(cost_y+cost_z+lambda_*cost_herd+lambda_*15*(s[0]+s[1]),GRB.MINIMIZE)

        # 
        #upper bound on x
        lm.addConstr(x.sum()<=limit_site)
        #test on the real case
# =============================================================================
#         lm.addConstr(x[10] == 1)
#         lm.addConstr(x[12] == 1)
#         lm.addConstr(x[23] == 1)
#         lm.addConstr(x[17] == 1)
#         lm.addConstr(x[3] == 1)
#         lm.addConstr(x[20] == 1)
# =============================================================================
        #smooth
        for t in range(2):
            for i in range(num_health_districts):
                for j in range(num_health_districts):
                    lm.addConstr(HL[i]-vv[t,i]-HL[j]+vv[t,j]<=s[t])
        for i in range(num_health_districts):
            #noncommuter get vaccinated
            lm.addConstr(y.sum(i, '*', '*')==totalpop[i]-np.sum(value,axis=1)[i])
            for t in range(time_periods):
                #constraints on vaccination
                lm.addConstr(vv[t,i]>=-gp.quicksum(y[i,j,t1] for j in range(num_health_districts) for t1 in range(t+1))
                             -gp.quicksum(z[i,j,k,t1] for j in range(num_health_districts) for k in range(num_health_districts) for t1 in range(t+1))
                             +HL[i])
                lm.addConstr(vv[t,i]>=0)
                lm.addConstr(y.sum( '*',i, t)+z.sum('*','*',i,t)<=400000)
                for j in range(num_health_districts):
                    lm.addConstr(y[i,j,t]<=400000*x[j])
                    lm.addConstr(y[i,j,t]>=0)
                    for k in range(num_health_districts):
                        lm.addConstr(z[i,j,k,t]<=400000*x[k])
                        lm.addConstr(z[i,j,k,t]>=0)
            for j in range(num_health_districts):
                #commuter get vaccinated
                lm.addConstr(z.sum(i, j,'*', '*')==value[i][j])
        # Optimize model
        #lm.setParam('TimeLimit', 10)
        lm.Params.Threads = 18
        lm.Params.OutputFlag = 1
        lm.Params.LogToConsole = 1
        lm.optimize()
        count=0
        print('LP:')
        for v in lm.getVars():
            if count<26:
                if v.X>0:
                    if v.VarName not in sol:
                        sol.append(v.VarName)
                    print('%s %g' % (v.VarName, v.X))
            count+=1
        print('Obj: %g' % lm.ObjVal)
        
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
vars_ = lm.getVars()
#for i in range(26):
#    print(vars_[25+26+i+1])
ans = 0
for v in vars_:
    if v.VarName[0]=='y':
        if v.VarName[len( v.VarName)-3:len( v.VarName)]=='14]':
            ans += v.X
            #print('%s %g' % (v.VarName, v.X))
    if v.VarName[0]=='z':
        if v.VarName[len( v.VarName)-3:len( v.VarName)]=='14]':
            ans += v.X
            #print('%s %g' % (v.VarName, v.X))
    if v.VarName[0]=='x':
        print('%s %f' % (v.VarName, v.X))
    if v.VarName[0]=='s':
        print('%s %f' % (v.VarName, v.X))
    if v.VarName[0]=='v':
        print('%s %f' % (v.VarName, v.X))
print(ans)
######
name = [];
loc_i = [];
loc_j = [];
loc_k = [];
loc_t = [];
value_sol = [];
real_v = np.zeros((6,26))
v_i = 0 
s0 = 0
s1 = 0
v_j = 0
s_i=0
for v in vars_:
    if v.VarName[0] == 'x':
        name.append(v.VarName[0])
        loc_i.append(int(v.VarName[2:-1].split(',')[0]))
        loc_j.append(0)
        loc_k.append(0)
        loc_t.append(0)
        value_sol.append(v.X)
    if v.VarName[0] == 'y':
        name.append(v.VarName[0])
        loc_i.append(int(v.VarName[2:-1].split(',')[0]))
        loc_j.append(int(v.VarName[2:-1].split(',')[1]))
        loc_k.append(0)
        loc_t.append(int(v.VarName[2:-1].split(',')[2]))
        value_sol.append(v.X)
    if v.VarName[0] == 'z':
        name.append(v.VarName[0])
        loc_i.append(int(v.VarName[2:-1].split(',')[0]))
        loc_j.append(int(v.VarName[2:-1].split(',')[1]))
        loc_k.append(int(v.VarName[2:-1].split(',')[2]))
        loc_t.append(int(v.VarName[2:-1].split(',')[3]))
        value_sol.append(v.X)
    if v.VarName[0] == 'v':
        real_v[v_i,v_j] = v.X
        if v_j ==25:
            v_j = 0
            v_i += 1
        else:
            v_j +=1
    if v.VarName[0] == 's':
        if s_i == 0:
            s0 = v.X
            s_i+=1
        else:
            s1 = v.X
        print(v.X)
end_time = time.time()
elapsed_time_seconds = end_time - start_time

# Convert elapsed time to hours
elapsed_time_hours = elapsed_time_seconds / 3600

print(f"Runtime: {elapsed_time_hours:.2f} hours")
# =============================================================================
# print('**************************')
# print(vaccinated_at_i2(real_v))
# print('**************************')
# =============================================================================

df = pd.DataFrame({'name': name, 'i': loc_i, 'j': loc_j, 'k': loc_k, 't': loc_t,'value':value_sol})
df.to_pickle('Results/base_od_4time_emp_10lambda_popweighted.pkl')
print('saved')
################################
#simple formulation
# =============================================================================
# limit_site  = 3
# sol = []
# for limit_site in range(1,28):
#     print('##########################################################################')
#     #for i in range(limit_site):
#     #    greedy_sol.append(np.where(weights ==sorted(weights)[num_health_districts-i-1])[0])
#     #    print(np.where(weights ==sorted(weights)[num_health_districts-i-1]))
#     
#     
#     def cost(x,y):
#         score = 0
#         for i in range(num_health_districts):
#             score += x[i]*totalpop[i]
#             for j in range(num_health_districts):
#                 score += y[i,j]*value[i][j]
#         return score
#     def sum_(x):
#         ans = 0
#         for i in range(num_health_districts):
#             ans += x[i]
#         return ans
#             
#     try:
#     
#         # Create a new model
#         lm = gp.Model("lm")
#     
#         # Create variables
#         
#         #this is number of patient assigned to empty beds from region i to region j
#         x = lm.addVars(num_health_districts,vtype=GRB.BINARY, name="x") 
#         #this is number of patient assigned to beds from delayed surgeries from region i to region j
#         y = lm.addVars(num_health_districts,num_health_districts,vtype=GRB.BINARY, name="y")
#         # Set objective
#         lm.setObjective(cost(x,y), GRB.MAXIMIZE)
#         #upper bound on x
#         lm.addConstr(sum_(x)<=limit_site)
#         #upper bound on y
#         for i in range(num_health_districts):
#             for j in range(num_health_districts):
#                 lm.addConstr(y[i,j]<=1-x[i])
#                 lm.addConstr(y[i,j]<=x[j])
#     
#         
#         # Optimize model
#         #lm.setParam('TimeLimit', 10)
#         lm.Params.LogToConsole = 0
#         lm.optimize()
#         count=0
#         print('LP:')
#         for v in lm.getVars():
#             if count<26:
#                 if v.X>0:
#                     if v.VarName not in sol:
#                         sol.append(v.VarName)
#                     print('%s %g' % (v.VarName, v.X))
#             count+=1
#     
#         print('Obj: %g' % lm.ObjVal)
#         
#     
#     except gp.GurobiError as e:
#         print('Error code ' + str(e.errno) + ': ' + str(e))
#     
#     except AttributeError:
#         print('Encountered an attribute error')
#     print('Greedy:')
# 
#     
# =============================================================================
