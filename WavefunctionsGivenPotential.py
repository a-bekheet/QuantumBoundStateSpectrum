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

def potential_energy(z, kappa):
    """
    Calculate the potential energy at a given point.
    
    :param z: Position
    :param kappa: Potential strength parameter
    :return: Potential energy at position z
    """
    if abs(z) < 1.0:
        return 0.0 ## WHERE YOU CHANGE POTENTIAL
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

# System parameters
length = 20.0
num_points = 2001
kappa = 2.0

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
selected_indices = sorted_indices[:4] ## WHERE YOU SELECT NUMBER OF PLOTS

# Extract energies and states
energies = eigenvalues[selected_indices]
states = eigenvectors[:, selected_indices]

# Plotting
fig, axs = plt.subplots(2 * len(selected_indices), figsize=(8, 14), sharex=True)

for i in range(len(selected_indices)):
    state = states[:, i]
    norm = m.sqrt(length / num_points * np.dot(state, state))
    state_normalized = state / norm
    
    # Plot probability density |ψ|^2
    axs[2 * i].set_ylabel('|ψ_{}|^2'.format(i))
    axs[2 * i].plot(positions, state_normalized**2, lw=2, label="$E_{}/\epsilon$={:.3f}".format(i, energies[i]))
    axs[2 * i].legend()

    # Plot real part of the wavefunction
    axs[2 * i + 1].set_ylabel('Re(ψ_{})'.format(i))
    axs[2 * i + 1].plot(positions, state_normalized.real, lw=2, color='green') # Including phase shift of pi

axs[-1].set_xlabel('z')