#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:44:26 2023

@author: suyanpengzhang
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import HPVnetwork as HPVN
import warnings
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB
import pickle
emp=np.array([0.9785489423063246,
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
0.9772144560727959])
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

totalpop = np.array([420697,443569,344450,526877,899111,339399,419797,308499,140361,547523,354750,479505,287613,666399,278815,193899,166374,379199,356465,195082,407864,321720,201739,411617,469439,465691])

# creating the dataset
# =============================================================================
# data = {}
# for i in range(26):
#     data[districts[i]] = emp[i]
# courses = list(data.keys())
# values = list(data.values())
#   
# fig = plt.figure(figsize = (8, 5),dpi=1200)
#  
# # creating the bar plot
# plt.bar(courses, values, color ='maroon', 
#         width = 0.4)
#  
# plt.xlabel("HD")
# plt.ylim([0.9, 1])
# plt.ylabel("Vaccination Target (%)")
# plt.title("Vaccination Target Across HDs")
# plt.tick_params(axis='x',direction='out',labelrotation=90)
# plt.show()
# fig.savefig('herd.pdf', format='pdf', dpi=1200, bbox_inches="tight")
# =============================================================================
