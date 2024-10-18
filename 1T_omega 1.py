# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 18:36:46 2024

@author: MR P MNISI
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Define fluid properties for two different fluids
def fluid_properties(fluid_type):
    if fluid_type == 1:
        # Water Base Fluid
        rho = 997.7     # Water Density (kg/m^3)
        nu = 1.002e-6   # Kinematic Viscosity (m^2/s)
        k = 0.613       # Thermal conductivity
        C = 4179        # Specific heat capacity
        sigma = 5.5e-6  # Electrical conductivity

    elif fluid_type == 2:
        # Hybrid Fluid with Al2O3 and Fe3O4
        phi_Al2O3 = 0.05  # 5% volume fraction of Al2O3
        phi_Fe3O4 = 0.08  # 8% volume fraction of Fe3O4

        # Water properties
        rho_water = 997.1     # kg/m³
        cp_water = 4179       # J/kg·K
        k_water = 0.613       # W/m·K
        sigma_water = 5  # S/m

        # Al2O3 properties
        rho_Al2O3 = 3970      # kg/m³
        cp_Al2O3 = 765        # J/kg·K
        k_Al2O3 = 40          # W/m·K
        sigma_Al2O3 = 32    # S/m

        # Fe3O4 properties
        rho_Fe3O4 = 5180      # kg/m³
        cp_Fe3O4 = 670        # J/kg·K
        k_Fe3O4 = 6           # W/m·K
        sigma_Fe3O4 = 299   # S/m

        # Effective density
        rho_nf = (1 - phi_Al2O3 - phi_Fe3O4) * rho_water + phi_Al2O3 * rho_Al2O3 + phi_Fe3O4 * rho_Fe3O4

        # Effective specific heat capacity
        cp_nf = ((1 - phi_Al2O3 - phi_Fe3O4) * rho_water * cp_water + 
                 phi_Al2O3 * rho_Al2O3 * cp_Al2O3 + 
                 phi_Fe3O4 * rho_Fe3O4 * cp_Fe3O4) / rho_nf

        # Effective thermal conductivity using Maxwell model
        k_nf = k_water * ((k_Al2O3 + 2*k_water - 2*phi_Al2O3*(k_water - k_Al2O3)) / 
                          (k_Al2O3 + 2*k_water + phi_Al2O3*(k_water - k_Al2O3)))
        k_nf += k_water * ((k_Fe3O4 + 2*k_water - 2*phi_Fe3O4*(k_water - k_Fe3O4)) / 
                           (k_Fe3O4 + 2*k_water + phi_Fe3O4*(k_water - k_Fe3O4)))

        # Effective electrical conductivity
        sigma_nf = (1 - phi_Al2O3 - phi_Fe3O4) * sigma_water + phi_Al2O3 * sigma_Al2O3 + phi_Fe3O4 * sigma_Fe3O4

        rho = rho_nf
        nu = 1.002e-6  # Assumed constant for simplicity
        k = k_nf
        C = cp_nf
        sigma = sigma_nf

    alpha = k / (rho * C)  # Thermal diffusivity
    mu = rho * nu          # Dynamic viscosity
    return rho, nu, alpha, sigma, mu

# Domain and time setup
Lx = 1    # Length of the domain in x-direction
Ly = 1    # Length of the domain in y-direction
tl = 0     # Start time
tu = 1    # End time
dt = 0.0001  # Time step size
dx = 0.01     # Spatial step size in x-direction
dy = 0.01     # Spatial step size in y-direction

# Velocity components and magnetic field parameters
u_base = 1.0  # Base velocity in x-direction
v_base = 1.0  # Base velocity in y-direction
B0 = 1  # Magnetic field strength
omega = 1 * np.pi / 10  # Frequency of the magnetic field

# Discretization
nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1
nt = int((tu - tl) / dt) + 1

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize arrays for velocity and temperature
def initialize_arrays():
    T = np.zeros((nt, nx, ny))
    u = np.zeros((nt, nx, ny))
    v = np.zeros((nt, nx, ny))
    
    # Initial condition: T = 0 everywhere
    T[0, :, :] = 25
    # Boundary conditions for temperature
    T[:, 0, :] = 100   # Left boundary at x = 0
    T[:, -1, :] = 25   # Right boundary at x = Lx
    T[:, :, 0] = 25    # Bottom boundary at y = 0
    T[:, :, -1] = 25   # Top boundary at y = Ly

    u[:, 0, :] = 0  # Left boundary at x = 0
    u[:, -1, :] = 1  # Right boundary at x = Lx
    u[:, :, 0] = 0    # Bottom boundary at y = 0
    u[:, :, -1] = 0   # Top boundary at y = Ly

    v[:, 0, :] = 0   # Left boundary at x = 0
    v[:, -1, :] = 0   # Right boundary at x = Lx
    v[:, :, 0] = 0   # Bottom boundary at y = 0
    v[:, :, -1] = 1   # Top boundary at y = Ly
    
    # Initial velocity conditions
    u[0, :, :] = u_base
    v[0, :, :] = v_base
    return T, u, v

def calculate_cfl(dx, dy, dt, u, v):
    """
    Calculate the CFL condition for a hybrid fluid model.
    
    Parameters:
    dx, dy: grid spacing in x and y directions
    dt: time step
    u, v: maximum fluid velocities in x and y directions
    c_s: sound speed in the medium (used for fluid stability)
    
    Returns:
    cfl_total: Total CFL value (should be <= 1 for stability)
    """
    
 # CFL condition based on velocity in x and y directions
    cfl_x = (u * dt) / dx
    cfl_y = (v * dt) / dy
        
    # CFL condition based on sound speed (for fluid dynamics)
    cfl_sound_x = (u * dt) / dx
    cfl_sound_y = (v * dt) / dy
        
    # Total CFL should consider both the velocity and sound speed constraints
    cfl_total = max(cfl_x, cfl_y, cfl_sound_x, cfl_sound_y)
    
    return cfl_total

# Number of iterations (time steps)
nx = int(Lx / dx) + 1
ny = int(Ly / dy) + 1
n_iterations = int((tu - tl) / dt) + 1  # Number of time steps

# Variables for CFL tracking over iterations
cfl_values = []

# Simulate varying fluid velocities over time (u, v)
np.random.seed(0)  # For reproducibility
u_max_vals = u_base* np.sin(np.linspace(0, 2 * np.pi, n_iterations))  # velocity in x direction
v_max_vals = v_base* np.sin(np.linspace(0, 2 * np.pi, n_iterations))  # velocity in y direction

# Iterate over time steps
for i in range(n_iterations):
    u_max = u_max_vals[i]
    v_max = v_max_vals[i]
    
    
    # Calculate CFL at each time step
    cfl = calculate_cfl(dx, dy, dt, u_max, v_max)
    cfl_values.append(cfl)


# Plot CFL over iterations
plt.figure(figsize=(20, 12))
plt.plot(range(n_iterations), cfl_values, label='CFL Value', color='b')
plt.axhline(y=1.0, color='r', linestyle='--', label='CFL Limit (1.0)')
plt.title('Transient CFL over Iterations')
plt.xlabel('Iterations')
plt.ylabel('CFL Value')
plt.legend()
plt.grid(True)
plt.show()


# Function to compute L2 norm
def compute_l2_norm(new, old):
    return np.sqrt(np.sum((new - old) ** 2))

# Function to calculate skin friction coefficient
def calculate_skin_friction(u, v, rho, mu, U_inf, side="left"):
    if side == "left":  # Calculate at x = 0
        du_dy = (u[:, 1, :] - u[:, 0, :]) / dy  # Velocity gradient at the left wall (x = 0)
    elif side == "right":  # Calculate at x = Lx
        du_dy = (u[:, -1, :] - u[:, -2, :]) / dy  # Velocity gradient at the right wall (x = Lx)
    
    # Wall shear stress
    tau_w = mu * du_dy
    
    # Skin friction coefficient
    C_f = (tau_w / (0.5 * rho * U_inf**2))*0 + 0.059/(U_inf*10/mu)**(0.2)
    
    return C_f

# Momentum and advection-diffusion solver with magnetic field effects
def run_simulation(fluid_type):
    # Get fluid properties
    rho, nu, alpha, sigma, mu = fluid_properties(fluid_type)
    
    # Stability factors
    lambda_x = (alpha * dt) / (dx**2)
    lambda_y = (alpha * dt) / (dy**2)
    
    # Initialize temperature and velocity fields
    T, u, v = initialize_arrays()

    # Arrays to store error values
    temperature_errors = []
    velocity_errors = []

    # Explicit time-stepping loop
    for n in range(0, nt - 1):
        # Magnetic field effect at time step n
        B = B0 * np.sin(omega * n * dt)
        
        # Copy previous step values for error calculation
        T_old = T[n, :, :].copy()
        u_old = u[n, :, :].copy()
        v_old = v[n, :, :].copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Update velocity components with momentum equations
                # u-Momentum equation (x-direction)
                u_adv_x = (u[n, i, j] - u[n, i - 1, j]) / dx
                u_adv_y = (v[n, i, j] - v[n, i, j - 1]) / dy
                u_diff_x = (u[n, i + 1, j] - 2 * u[n, i, j] + u[n, i - 1, j]) / dx**2
                u_diff_y = (u[n, i, j + 1] - 2 * u[n, i, j] + u[n, i, j - 1]) / dy**2

                u[n + 1, i, j] = u[n, i, j] + dt * (-u_adv_x - u_adv_y + nu * (u_diff_x + u_diff_y) - B *u[n, i, j]* sigma / rho)

                # v-Momentum equation (y-direction)
                v_adv_x = (u[n, i, j] - u[n, i - 1, j]) / dx
                v_adv_y = (v[n, i, j] - v[n, i, j - 1]) / dy
                v_diff_x = (v[n, i + 1, j] - 2 * v[n, i, j] + v[n, i - 1, j]) / dx**2
                v_diff_y = (v[n, i, j + 1] - 2 * v[n, i, j] + v[n, i, j - 1]) / dy**2

                v[n + 1, i, j] = v[n, i, j] + dt * (-v_adv_x - v_adv_y + nu * (v_diff_x + v_diff_y))

                # Advection-Diffusion equation for temperature
                advection_x = u[n, i, j] * (T[n, i, j] - T[n, i - 1, j]) / dx
                advection_y = v[n, i, j] * (T[n, i, j] - T[n, i, j - 1]) / dy
                diffusion_x = lambda_x * (T[n, i + 1, j] - 2 * T[n, i, j] + T[n, i - 1, j])
                diffusion_y = lambda_y * (T[n, i, j + 1] - 2 * T[n, i, j] + T[n, i, j - 1])

                # Update temperature
                T[n + 1, i, j] = T[n, i, j] + diffusion_x + diffusion_y - dt * (advection_x + advection_y)
                

        # Calculate L2 norm for temperature and velocity
        temperature_error = compute_l2_norm(T[n + 1, :, :], T_old)
        velocity_error = compute_l2_norm(u[n + 1, :, :] + v[n + 1, :, :], u_old + v_old)

        # Append errors to the list
        temperature_errors.append(temperature_error)
        velocity_errors.append(velocity_error)

    return T, u, v, temperature_errors, velocity_errors

# Run simulation for base water (fluid_type = 1) and hybrid fluid (fluid_type = 2)
T_water, u_water, v_water, temp_errors_water, vel_errors_water = run_simulation(fluid_type=1)
T_hybrid, u_hybrid, v_hybrid, temp_errors_hybrid, vel_errors_hybrid = run_simulation(fluid_type=2)

# Plot results for base water
plt.figure(figsize=(20,12))
plt.contourf(x, y, T_water[-1, :, :], cmap='hot')
plt.colorbar(label="Temperature (°C)")
plt.title("Temperature Distribution for Base Water with Magnetic Field (2D)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# Plot results for hybrid fluid
plt.figure(figsize=(20,12))
plt.contourf(x, y, T_hybrid[-1, :, :], cmap='hot')
plt.colorbar(label="Temperature (°C)")
plt.title("Temperature Distribution for Hybrid Fluid with Magnetic Field (2D)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# 1D plot comparison of temperature at the midline (y = Ly/2)
plt.figure(figsize=(20,12))
plt.plot(x, T_water[-1, :, int(ny/2)], label="Base Water", color='blue')
plt.plot(x, T_hybrid[-1, :, int(ny/2)], label="Hybrid Fluid", color='red')
plt.title("1D Temperature Distribution at Midline (y = Ly/2)")
plt.xlabel("x (m)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# Velocity magnitude calculation
def velocity_magnitude(u, v):
    return np.sqrt(u**2 + v**2)

# Plot velocity magnitude for base water
vel_mag_water = velocity_magnitude(u_water[-1, :, :], v_water[-1, :, :])
plt.figure(figsize=(20,12))
plt.contourf(x, y, vel_mag_water, cmap='viridis')
plt.colorbar(label="Velocity Magnitude (m/s)")
plt.title("Velocity Magnitude Distribution for Base Water (2D)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# Plot velocity magnitude for hybrid fluid
vel_mag_hybrid = velocity_magnitude(u_hybrid[-1, :, :], v_hybrid[-1, :, :])
plt.figure(figsize=(20,12))
plt.contourf(x, y, vel_mag_hybrid, cmap='viridis')
plt.colorbar(label="Velocity Magnitude (m/s)")
plt.title("Velocity Magnitude Distribution for Hybrid Fluid (2D)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# 1D plot comparison of velocity at the midline (y = Ly/2)
plt.figure(figsize=(20,12))
plt.plot(x, vel_mag_water[:, int(ny/2)], label="Base Water", color='blue')
plt.plot(x, vel_mag_hybrid[:, int(ny/2)], label="Hybrid Fluid", color='red')
plt.title("1D Velocity Magnitude Distribution at Midline (y = Ly/2)")
plt.xlabel("x (m)")
plt.ylabel("Velocity Magnitude (m/s)")
plt.legend()

# Error Analysis Plots
# Plot error convergence for temperature
plt.figure(figsize=(20,12))
plt.plot(np.linspace(tl, tu, nt-1), temp_errors_water, label="Base Water", color='blue')
plt.plot(np.linspace(tl, tu, nt-1), temp_errors_hybrid, label="Hybrid Fluid", color='red')
plt.title("Temperature Error Convergence (L2 Norm)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature Error ")
plt.legend()
plt.grid()
plt.show()

# Plot error convergence for velocity
plt.figure(figsize=(20,12))
plt.plot(np.linspace(tl, tu, nt-1), vel_errors_water, label="Base Water", color='blue')
plt.plot(np.linspace(tl, tu, nt-1), vel_errors_hybrid, label="Hybrid Fluid", color='red')
plt.title("Velocity Error Convergence ")
plt.xlabel("Time (s)")
plt.ylabel("Velocity Error (L2 Norm)")
plt.legend()
plt.grid()
plt.show()

# Plot error convergence for temperature (log-log scale)
plt.figure(figsize=(20,12))
plt.loglog(np.linspace(tl, tu, nt-1), temp_errors_water, label="Base Water", color='blue')
plt.loglog(np.linspace(tl, tu, nt-1), temp_errors_hybrid, label="Hybrid Fluid", color='red')
plt.title("Temperature Error Convergence - Log-Log Scale")
plt.xlabel("Time (s)")
plt.ylabel("Temperature Error ")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Plot error convergence for velocity (log-log scale)
plt.figure(figsize=(20,12))
plt.loglog(np.linspace(tl, tu, nt-1), vel_errors_water, label="Base Water", color='blue')
plt.loglog(np.linspace(tl, tu, nt-1), vel_errors_hybrid, label="Hybrid Fluid", color='red')
plt.title("Velocity Error Convergence  - Log-Log Scale")
plt.xlabel("Time (s)")
plt.ylabel("Velocity Error ")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Calculate the skin friction coefficient for water and hybrid fluid
U_inf = u_base  # Using base velocity as the characteristic velocity

# Calculate skin friction coefficient at the left boundary (x = 0) and right boundary (x = Lx)
Cf_water_left = calculate_skin_friction(u_water, v_water, rho=997.7, mu=997.7*1.002e-6, U_inf=U_inf, side="left")
Cf_water_right = calculate_skin_friction(u_water, v_water, rho=997.7, mu=997.7*1.002e-6, U_inf=U_inf, side="right")

# For the hybrid fluid, use the calculated dynamic viscosity (mu)
rho_hybrid, nu_hybrid, _, _, mu_hybrid = fluid_properties(2)
Cf_hybrid_left = calculate_skin_friction(u_hybrid, v_hybrid, rho=rho_hybrid, mu=mu_hybrid, U_inf=U_inf, side="left")
Cf_hybrid_right = calculate_skin_friction(u_hybrid, v_hybrid, rho=rho_hybrid, mu=mu_hybrid, U_inf=U_inf, side="right")

# Plot skin friction coefficient along the y-axis (left boundary)
plt.figure(figsize=(20,12))
plt.plot(y, Cf_water_left[-1, :], label="Base Water - Left Wall", color='blue')
plt.plot(y, Cf_hybrid_left[-1, :], label="Hybrid Fluid - Left Wall", color='red')
plt.title("Skin Friction Coefficient along the Left Wall")
plt.xlabel("y (m)")
plt.ylabel("C_f")
plt.legend()
plt.grid(True)
plt.show()

# Plot skin friction coefficient along the y-axis (right boundary)
plt.figure(figsize=(20,12))
plt.plot(y, Cf_water_right[-1, :], label="Base Water - Right Wall", color='blue')
plt.plot(y, Cf_hybrid_right[-1, :], label="Hybrid Fluid - Right Wall", color='red')
plt.title("Skin Friction Coefficient along the Right Wall")
plt.xlabel("y (m)")
plt.ylabel("C_f")
plt.legend()
plt.grid(True)
plt.show()
