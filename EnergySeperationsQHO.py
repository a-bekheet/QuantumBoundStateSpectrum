#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:56:59 2023

@author: bekheet
"""
import numpy as np
import math as m
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Conversion factor from Hartrees to eV
hartree_to_ev = 27.2114

# System parameters
length = 1 # Length in Bohr radii (a0)
num_points = 1001 # Number of discretization points (unitless)
kappa = 1 * hartree_to_ev  # kappa in eV/a0^2

def potential_energy(z, kappa):
    """
    Calculate the potential energy at a given point.
    
    :param z: Position
    :param kappa: Potential strength parameter
    :return: Potential energy at position z
    """
    if abs(z) < 0.05:
        return 0.5 * kappa * z**2 ## WHERE YOU CHANGE POTENTIAL
    else:
        return 0.0

def potential_matrix_element(row, col, kappa):
    """
    Compute the potential energy matrix element.
    
    :param row: Row index in the matrix
    :param col: Column index in the matrix
    :param kappa: Potential strength parameter
    :return: Matrix element of potential energy
    """
    if row == col:
        return potential_energy(positions[row], kappa)
    return 0.0

def kinetic_matrix_element(row, col, num_points, length):
    """
    Compute the kinetic energy matrix element using finite difference method.
    
    :param row: Row index in the matrix
    :param col: Column index in the matrix
    :param num_points: Number of discretization points
    :param length: Length of the system
    :return: Matrix element of kinetic energy
    """
    factor = -num_points**2 / length**2
    if row == col:
        return -2 * factor
    elif abs(row - col) == 1:
        return factor
    else:
        return 0.0

# Generate position array
positions = np.linspace(-length / 2, length / 2, num_points)

# Initialize Hamiltonian, potential, and kinetic energy matrices
hamiltonian = np.zeros((num_points, num_points))
potential_energy_matrix = np.zeros((num_points, num_points))
kinetic_energy_matrix = np.zeros((num_points, num_points))

# Populate potential and kinetic energy matrices
for row in range(num_points):
    for col in range(num_points):
        potential_energy_matrix[row, col] = potential_matrix_element(row, col, kappa)
        kinetic_energy_matrix[row, col] = kinetic_matrix_element(row, col, num_points, length)

# Calculate the Hamiltonian
hamiltonian = kinetic_energy_matrix + potential_energy_matrix

# Plot potential energy
plt.plot(positions, [potential_energy(z, kappa) for z in positions])

# Solve the eigenvalue problem
eigenvalues, eigenvectors = LA.eig(hamiltonian)

# Sort and select the eigenvalues
sorted_indices = eigenvalues.argsort()
selected_indices = sorted_indices[:10] ## WHERE YOU SELECT NUMBER OF PLOTS

# Extract energies and states
energies = eigenvalues[selected_indices]
states = eigenvectors[:, selected_indices]

# Plotting: Create two subplots, one for probability densities and one for real parts
fig, axs = plt.subplots(2, figsize=(8, 6), sharex=True)

# Set labels for subplots
axs[0].set_ylabel('|ψ|^2')
axs[1].set_ylabel('Re(ψ)')
axs[-1].set_xlabel('z')

# Plot each state's data on the respective subplot
for i in range(len(selected_indices)):
    state = states[:, i]
    norm = m.sqrt(length / num_points * np.dot(state, state))
    state_normalized = state / norm
    
    # Plot probability density |ψ_i|^2
    axs[0].plot(positions, state_normalized**2, lw=2, label="State {} ($E_{}/\epsilon$={:.3f})".format(i, i, energies[i]))

    # Plot real part of the wavefunction Re(ψ_i)
    axs[1].plot(positions, state_normalized.real, lw=2, label="State {}".format(i))

# Add legends to subplots
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.show()

# Function to fit the data
def linear_fit(x, m, c):
    return m * x + c

# Calculate the energy spacings for the first 10 states
num_states_to_analyze = 10
energy_spacings = np.diff(energies[:num_states_to_analyze])

# Generate state indices for x-axis (should be one less than num_states_to_analyze)
state_indices = np.arange(1, num_states_to_analyze)

# Ensure the lengths of state_indices and energy_spacings match
state_indices = state_indices[:len(energy_spacings)]

# Perform the linear fit
params, covariance = curve_fit(linear_fit, state_indices, energy_spacings)
m, c = params

# Plot the energy spacings
plt.figure(figsize=(8, 4))
plt.plot(state_indices, energy_spacings, 'o', label='Energy Spacings')
plt.plot(state_indices, linear_fit(state_indices, m, c), label=f'Fit: y = {m:.2f}x + {c:.2f}')
plt.xlabel('State Index')
plt.ylabel('Energy Spacing')
plt.title('Energy Spacing Between Successive States - Quantum Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()

# Print the gradient and intercept
print(f"Gradient (Rate of Change): {m:.5f}")
print(f"Intercept: {c:.5f}")