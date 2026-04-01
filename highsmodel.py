import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import pickle
from pyomo.contrib.appsi.solvers.highs import Highs

# Initialize the persistent solver
import time

# --- DATA LOADING ---
start_time = time.time()
num_health_districts = 26
time_periods = 6

# Use your existing paths
base_path = '/Users/suyanpengzhang/Documents/GitHub.nosync/Vaccination-Allocation/Data/'
od_flow = pd.read_csv(base_path + 'mean_df_20210224_20210424.csv')

with open(base_path + "travel_time.pkl", "rb") as file:
    c_m_v = pickle.load(file)

c_matrix = np.array([[float(i) for i in j] for j in c_m_v])
value = od_flow.values[0:-1, 1:-1].astype(float)
actualcommunter_scaler = 4753898 / np.sum(value)
value = (value * actualcommunter_scaler).astype(int)

totalpop = [420697,443569,344450,526877,899111,339399,419797,308499,140361,547523,354750,479505,287613,666399,278815,193899,166374,379199,356465,195082,407864,321720,201739,411617,469439,465691]
emp = [0.9785489423063246, 0.9749523393023726, 0.9810134958440276, 0.9715990029226316, 0.9512510687291531, 0.9793683522808072, 0.9739319327227332, 0.9817156078851325, 0.9915252477779021, 0.9678482424468472, 0.9760508448195166, 0.9657752125951641, 0.9784091897580796, 0.967984322913777, 0.9795198849727248, 0.9859672856449567, 0.9874863827092342, 0.9723956829474802, 0.9745472149882571, 0.985468457303294, 0.9697058020586842, 0.9771345654416476, 0.9898114604005365, 0.979337971566636, 0.9766137484373549, 0.9772144560727959]
HL = np.array(totalpop) * np.array(emp)

# Pre-calculate D matrix
D = np.zeros((26, 26, 26))
for i in range(26):
    for j in range(26):
        for k in range(26):
            D[i,j,k] = max(0, min(c_matrix[i][k]+c_matrix[k][i], c_matrix[j][k]+c_matrix[k][j], 
                                  c_matrix[j][k]+c_matrix[k][i]-c_matrix[j][i], c_matrix[i][k]+c_matrix[k][j]-c_matrix[i][j]))

weights_bc = 50 * np.sum(value, axis=1) / np.sum(value)
weights = 50 * weights_bc / np.sum(weights_bc)
lambda_ = 9
lambda1_ = 150 * np.mean(totalpop)

# --- MODEL BUILDING ---
model = pyo.ConcreteModel()

# Dimensions
I = range(num_health_districts)
T = range(time_periods)

# Variables
model.x = pyo.Var(I, domain=pyo.Binary)
model.y = pyo.Var(I, I, T, domain=pyo.NonNegativeIntegers)
model.z = pyo.Var(I, I, I, T, domain=pyo.NonNegativeIntegers)
model.v = pyo.Var(T, I, domain=pyo.NonNegativeReals)
model.s = pyo.Var(range(8), domain=pyo.Reals)

# Objective
def objective_rule(m):
    cost_y = sum(m.y[i, j, t] * (2 * c_matrix[i][j]) for i in I for j in I for t in T)
    cost_z = sum(m.z[i, j, k, t] * D[i, j, k] for i in I for j in I for k in I for t in T)
    cost_herd = sum(m.v[t, i] * weights[i] * (0.9**t) for t in T for i in I)
    cost_smooth = sum(m.s[idx] for idx in range(2)) # Aligning with your loop range(6)
    return cost_y + cost_z + 0.25 * lambda_ * cost_herd + 0.25 * lambda1_ * cost_smooth

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints
model.con_budget = pyo.Constraint(expr=sum(model.x[i] for i in I) <= 6)

# Fixing original sites
for site in [10, 6, 3, 4, 18, 23]:
    model.x[site].fix(1)

# Smoothness constraints
def smooth_rule(m, t, i, j):
    lhs = (sum(m.y[i, i1, t1] for i1 in I for t1 in range(t+1)) + 
           sum(m.z[i, i1, i2, t1] for i1 in I for i2 in I for t1 in range(t+1))) / totalpop[i]
    rhs = (sum(m.y[j, i1, t1] for i1 in I for t1 in range(t+1)) + 
           sum(m.z[j, i1, i2, t1] for i1 in I for i2 in I for t1 in range(t+1))) / totalpop[j]
    return lhs - rhs <= m.s[t]

model.con_smooth = pyo.Constraint(range(6), I, I, rule=smooth_rule)

# Population Coverage
def noncommuter_rule(m, i):
    return sum(m.y[i, j, t] for j in I for t in T) == totalpop[i] - np.sum(value, axis=1)[i]
model.con_noncommuter = pyo.Constraint(I, rule=noncommuter_rule)

def commuter_rule(m, i, j):
    return sum(m.z[i, j, k, t] for k in I for t in T) == value[i][j]
model.con_commuter = pyo.Constraint(I, I, rule=commuter_rule)

# Activation & Capacity
def y_act(m, i, j, t):
    return m.y[i, j, t] <= 100000 * m.x[j]
model.con_y_act = pyo.Constraint(I, I, T, rule=y_act)

def z_act(m, i, j, k, t):
    return m.z[i, j, k, t] <= 100000 * m.x[k]
model.con_z_act = pyo.Constraint(I, I, I, T, rule=z_act)

def v_bound(m, t, i):
    return m.v[t, i] >= HL[i] - sum(m.y[i, j, t1] for j in I for t1 in range(t+1)) - \
           sum(m.z[i, j, k, t1] for j in I for k in I for t1 in range(t+1))
model.con_v_bound = pyo.Constraint(T, I, rule=v_bound)

# --- SOLVING ---
# Use 'highspy' which is the Python interface for HiGHS
solver = Highs()
print("Building complete. Starting HiGHS Solver...")
solver.solve(model, tee=True)

# --- RESULTS ---
print(f"Final Objective: {pyo.value(model.obj)}")
for i in I:
    if pyo.value(model.x[i]) > 0.5:
        print(f"Selected District: {i}")

# Saving results (matching your original format)
res_data = []
for v in model.component_objects(pyo.Var, active=True):
    for index in v:
        res_data.append({'name': v.name, 'index': index, 'value': pyo.value(v[index])})

# Reformatting to match your P2_weekly.pkl structure
# You can further refine this to match your exact loc_i, loc_j logic
df_res = pd.DataFrame(res_data)
df_res.to_pickle('/Users/suyanpengzhang/Documents/GitHub.nosync/Vaccination-Allocation/RevisedResults/P2_weekly_highs.pkl')

print(f"Runtime: {(time.time() - start_time)/3600:.2f} hours")