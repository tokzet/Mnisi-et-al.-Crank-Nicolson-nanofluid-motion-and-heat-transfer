# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 18:36:46 2024

@author: MR P MNISI
"""

import numpy as np
import matplotlib.pyplot as plt


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
        sigma_water = 5.5e-6  # S/m

        # Al2O3 properties
        rho_Al2O3 = 3970      # kg/m³
        cp_Al2O3 = 765        # J/kg·K
        k_Al2O3 = 40          # W/m·K
        sigma_Al2O3 = 35e6    # S/m

        # Fe3O4 properties
        rho_Fe3O4 = 5180      # kg/m³
        cp_Fe3O4 = 670        # J/kg·K
        k_Fe3O4 = 6           # W/m·K
        sigma_Fe3O4 = 25000   # S/m

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
dt = 0.001  # Time step size
dx = 0.01     # Spatial step size in x-direction
dy = 0.01     # Spatial step size in y-direction

# Velocity components and magnetic field parameters
u_base = 1.0  # Base velocity in x-direction
v_base = 1.0  # Base velocity in y-direction
B0 = 1  # Magnetic field strength
omega = 1 * np.pi  # Frequency of the oscillating magnetic field

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

# CFL condition calculation
def calculate_cfl(dx, dy, dt, u, v, alpha):
    cfl_x = np.max(np.abs(u)) * dt / dx
    cfl_y = np.max(np.abs(v)) * dt / dy
    cfl_diffusion = alpha * dt / (min(dx, dy)**2)
    return max(cfl_x, cfl_y, cfl_diffusion)

# Momentum and temperature equation solver with CFL tracking
def solve_equations_with_cfl(T, u, v, rho, nu, alpha, sigma, mu):
    # Stability factors
    lambda_x = alpha * dt / (dx**2)
    lambda_y = alpha * dt / (dy**2)

    cfl_values = []
    temperature_errors = []
    velocity_errors = []

    for n in range(nt - 1):
        t = n * dt
        B = B0 * np.sin(omega * t)  # Oscillating magnetic field

        cfl = calculate_cfl(dx, dy, dt, u[n], v[n], alpha)
        cfl_values.append(cfl)

        T_old = T[n, :, :]
        u_old = u[n, :, :]
        v_old = v[n, :, :]

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Temperature equation (explicit update)
                T[n + 1, i, j] = T[n, i, j] + lambda_x * (T[n, i+1, j] - 2*T[n, i, j] + T[n, i-1, j]) + \
                                 lambda_y * (T[n, i, j+1] - 2*T[n, i, j] + T[n, i, j-1]) - \
                                 dt * (u[n, i, j] * (T[n, i, j] - T[n, i-1, j]) / dx + \
                                       v[n, i, j] * (T[n, i, j] - T[n, i, j-1]) / dy)

                # x-Momentum equation
                u_adv_x = u[n, i, j] * (u[n, i, j] - u[n, i-1, j]) / dx
                u_adv_y = v[n, i, j] * (u[n, i, j] - u[n, i, j-1]) / dy
                u_diff_x = nu * (u[n, i+1, j] - 2*u[n, i, j] + u[n, i-1, j]) / dx**2
                u_diff_y = nu * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1]) / dy**2
                u[n+1, i, j] = u[n, i, j] + dt * (-u_adv_x - u_adv_y + u_diff_x + u_diff_y - sigma * B**2 * u[n, i, j] / rho)

                # y-Momentum equation
                v_adv_x = u[n, i, j] * (v[n, i, j] - v[n, i-1, j]) / dx
                v_adv_y = v[n, i, j] * (v[n, i, j] - v[n, i, j-1]) / dy
                v_diff_x = nu * (v[n, i+1, j] - 2*v[n, i, j] + v[n, i-1, j]) / dx**2
                v_diff_y = nu * (v[n, i, j+1] - 2*v[n, i, j] + v[n, i, j-1]) / dy**2
                v[n+1, i, j] = v[n, i, j] + dt * (-v_adv_x - v_adv_y + v_diff_x + v_diff_y)

        temperature_errors.append(np.linalg.norm(T[n + 1, :, :] - T_old))
        velocity_errors.append(np.linalg.norm(u[n + 1, :, :] + v[n + 1, :, :] - (u_old + v_old)))

    return T, u, v, cfl_values, temperature_errors, velocity_errors

# Run simulation for water
rho1, nu1, alpha1, sigma1, mu1 = fluid_properties(1)
T1, u1, v1 = initialize_arrays()
T1, u1, v1, cfl_values1, temp_errors1, vel_errors1 = solve_equations_with_cfl(T1, u1, v1, rho1, nu1, alpha1, sigma1, mu1)

# Run simulation for hybrid fluid
rho2, nu2, alpha2, sigma2, mu2 = fluid_properties(2)
T2, u2, v2 = initialize_arrays()
T2, u2, v2, cfl_values2, temp_errors2, vel_errors2 = solve_equations_with_cfl(T2, u2, v2, rho2, nu2, alpha2, sigma2, mu2)

# Plot CFL values for stability analysis
plt.figure(figsize=(12, 8))
plt.plot(range(nt - 1), cfl_values1, label='Water', color='blue')
plt.plot(range(nt - 1), cfl_values2, label='Hybrid Fluid', color='red')
plt.axhline(1.0, color='black', linestyle='--', label='CFL Limit (1.0)')
plt.title('CFL Stability Analysis Over Time')
plt.xlabel('Time Steps')
plt.ylabel('CFL Value')
plt.legend()
plt.grid()
plt.show()

# Error convergence plot for temperature (log-log scale)
plt.figure(figsize=(12, 8))
plt.loglog(range(nt - 1), temp_errors1, label='Water Temperature Error', color='blue')
plt.loglog(range(nt - 1), temp_errors2, label='Hybrid Fluid Temperature Error', color='red')
plt.title('Temperature Error Convergence (Log-Log Scale)')
plt.xlabel('Time Steps')
plt.ylabel('L2 Norm of Temperature Error')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()

# Error convergence plot for velocity (log-log scale)
plt.figure(figsize=(12, 8))
plt.loglog(range(nt - 1), vel_errors1, label='Water Velocity Error', color='blue')
plt.loglog(range(nt - 1), vel_errors2, label='Hybrid Fluid Velocity Error', color='red')
plt.title('Velocity Error Convergence (Log-Log Scale)')
plt.xlabel('Time Steps')
plt.ylabel('L2 Norm of Velocity Error')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()

# Plot velocity distribution for Water (2D)
plt.figure(figsize=(12, 8))
plt.contourf(x, y, (u1[-1, :, :]**2 + v1[-1, :, :]**2)**0.5, cmap='viridis')
plt.colorbar(label='Velocity Magnitude (m/s)')
plt.title('Velocity Field for Water')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid()
plt.show()

# Plot velocity distribution for Hybrid Fluid (2D)
plt.figure(figsize=(12, 8))
plt.contourf(x, y, (u2[-1, :, :]**2 + v2[-1, :, :]**2)**0.5, cmap='viridis')
plt.colorbar(label='Velocity Magnitude (m/s)')
plt.title('Velocity Field for Hybrid Fluid')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid()
plt.show()

# 1D Velocity Plot: Midline Comparison for Water and Hybrid Fluid (u and v)
mid_y = int(ny / 2)  # Index for midline along y-direction

plt.figure(figsize=(12, 8))
plt.plot(x, u1[-1, :, mid_y], label='Water - u', color='blue', linestyle='--')
plt.plot(x, u2[-1, :, mid_y], label='Hybrid Fluid - u', color='red')
plt.plot(x, v1[-1, :, mid_y], label='Water - v', color='cyan', linestyle='--')
plt.plot(x, v2[-1, :, mid_y], label='Hybrid Fluid - v', color='orange')
plt.title('1D Velocity Components at Midline (y = Ly/2)')
plt.xlabel('x (m)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()
plt.show()

# Error check and plot for temperature comparison
errors = []
time_steps = np.arange(1, nt)

for n in time_steps:
    error = np.linalg.norm(T1[n, :, :] - T2[n, :, :])
    errors.append(error)

plt.figure(figsize=(12, 8))
plt.plot(time_steps * dt, errors, label='Temperature Error (Water vs Hybrid Fluid)', color='purple')
plt.title('Temperature Difference Over Time')
plt.xlabel('Time (s)')
plt.ylabel('L2 Norm of Temperature Error')
plt.legend()
plt.grid()
plt.show()

# Visualization of temperature distribution for Water
plt.figure(figsize=(10, 8))
plt.contourf(x, y, T1[-1, :, :], cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution for Water')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# Visualization of temperature distribution for Hybrid Fluid
plt.figure(figsize=(10, 8))
plt.contourf(x, y, T2[-1, :, :], cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution for Hybrid Fluid')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# 1D Temperature Plot: Midline Comparison between Water and Hybrid Fluid
mid_y = int(ny / 2)  # Index for midline along y-direction

plt.figure(figsize=(12, 8))
plt.plot(x, T1[-1, :, mid_y], label='Water', color='blue', linestyle='--')
plt.plot(x, T2[-1, :, mid_y], label='Hybrid Fluid', color='red')
plt.title('1D Temperature Distribution at Midline (y = Ly/2)')
plt.xlabel('x (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.show()
