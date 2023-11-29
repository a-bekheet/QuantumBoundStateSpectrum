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

alpha = 3 # value of alpha for the potential of form abs(x)**alpha
num_states_to_analyze = 6 # number of states to investigate

# System parameters
length = 12 # Length in Bohr radii (a0)
num_points = 1001 # Number of discretization points (unitless)
kappa = 1 * hartree_to_ev  # kappa in eV/a0^2

def potential_energy(x, alpha):
    """
    Calculate the potential energy for a given x using the form |x|^alpha.
    
    :param x: Position
    :param alpha: Exponent in the potential function, with alpha >= 2
    :return: Potential energy at position x
    """
    if alpha < 2:
        raise ValueError("Alpha must be greater than or equal to 2")

    return abs(x)**alpha

def potential_matrix_element(row, col, alpha):
    """
    Compute the potential energy matrix element using U(x) = |x|^alpha.
    
    :param row: Row index in the matrix
    :param col: Column index in the matrix
    :param alpha: Exponent in the potential function
    :return: Matrix element of potential energy
    """
    if row == col:
        return potential_energy(positions[row], alpha)
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
        potential_energy_matrix[row, col] = potential_matrix_element(row, col, alpha)
        kinetic_energy_matrix[row, col] = kinetic_matrix_element(row, col, num_points, length)

# Calculate the Hamiltonian
hamiltonian = kinetic_energy_matrix + potential_energy_matrix

# Plot potential energy
plt.plot(positions, [potential_energy(z, kappa) for z in positions])

# Solve the eigenvalue problem
eigenvalues, eigenvectors = LA.eig(hamiltonian)

# Sort and select the eigenvalues
sorted_indices = eigenvalues.argsort()
selected_indices = sorted_indices[:num_states_to_analyze]

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

# Calculate the energy spacings for the first N states
energy_spacings = np.diff(energies[:num_states_to_analyze])

# Ensure that the length of the energies array matches num_states_to_analyze
if len(energies) < num_states_to_analyze:
    raise ValueError(f"Expected at least {num_states_to_analyze} energy levels, but got {len(energies)}.")

# Generate state indices for x-axis
state_indices = np.arange(1, num_states_to_analyze + 1)  # +1 because it's inclusive


# Function to fit the data quadratically
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c


# State indices for the energy levels
state_indices = np.arange(1, num_states_to_analyze + 1)  # +1 because it's inclusive

# Perform the quadratic fit on the energy levels
params, covariance = curve_fit(quadratic_fit, state_indices, energies[:num_states_to_analyze])
a, b, c = params

# Plot the energy levels and the quadratic fit
plt.figure(figsize=(8, 4))
plt.plot(state_indices, energies[:num_states_to_analyze], 'o', label='Energy Levels')
plt.plot(state_indices, quadratic_fit(state_indices, a, b, c), label=f'Quadratic Fit: y = {a:.2f}x² + {b:.2f}x + {c:.2f}')
plt.xlabel('State Index')
plt.ylabel('Energy')
plt.title('Energy Levels vs. State Index - Bound State Spectrum')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients of the quadratic fit
print(f"Coefficient of x² (a): {a:.5f}")
print(f"Coefficient of x (b): {b:.5f}")
print(f"Intercept (c): {c:.5f}")
