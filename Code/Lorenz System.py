import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system equations
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 10
beta = 8 / 3
rho_values = [10, 28, 50]  # Example rho values to study
initial_state = [1, 1, 1]  # Initial condition
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Simulations for different rho values
for rho in rho_values:
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=(sigma, beta, rho))
    x, y, z = solution.y

    # 3D Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_title(f"Lorenz Attractor (rho={rho})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
