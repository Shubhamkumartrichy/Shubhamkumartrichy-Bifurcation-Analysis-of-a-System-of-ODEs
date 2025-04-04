import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    return r * x * (1 - x)

r_values = np.linspace(0, 4, 1000)  
iterations = 1000                   
last = 100                          

x = 1e-5 * np.ones(len(r_values))   
results = []

for _ in range(iterations):
    x = logistic_map(r_values, x)   
    if _ >= (iterations - last):    
        results.append(np.copy(x))

results = np.array(results).flatten()
r_repeated = np.tile(r_values, last)  

plt.figure(figsize=(12, 6))
plt.plot(r_repeated, results, ',k', alpha=0.25)
plt.title("Bifurcation Diagram of the Logistic Map")
plt.xlabel("Growth Rate (r)")
plt.ylabel("Population (x)")
plt.grid(alpha=0.3)
plt.show()
