import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

sigma = 10.0
beta = 8.0 / 3.0
rho_values = [10, 28, 50]  
initial_state = [1.0, 1.0, 1.0]  
t_span = (0, 50)  
t_eval = np.linspace(t_span[0], t_span[1], 10000)  

for rho in rho_values:
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=(sigma, beta, rho))
    x, y, z = solution.y

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, lw=0.5)
    ax.set_title(f"Lorenz System (rho={rho})")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()

def bifurcation_lorenz(rho_start, rho_end, steps):
    bifurcation_points = []
    rho_values = np.linspace(rho_start, rho_end, steps)
    for rho in rho_values:
        solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=(sigma, beta, rho))
        z_values = solution.y[2][-500:]  
        bifurcation_points.extend([(rho, z) for z in z_values])
    return bifurcation_points

rho_start = 0
rho_end = 50
steps = 200
bifurcation_data = bifurcation_lorenz(rho_start, rho_end, steps)
rho_bifurcation, z_bifurcation = zip(*bifurcation_data)

plt.figure(figsize=(12, 6))
plt.scatter(rho_bifurcation, z_bifurcation, s=0.5, color="black")
plt.title("Bifurcation Diagram of the Lorenz System")
plt.xlabel("Rho (Control Parameter)")
plt.ylabel("Z (State Variable)")
plt.grid()
plt.show()
