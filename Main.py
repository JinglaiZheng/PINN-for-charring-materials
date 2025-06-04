# -*- coding: utf-8 -*-

"""
File: Main.py
Author: Jing-lai Zheng
Date: 2024-04-01
Description: This is the code for loading the pre-trained PINN model and making predictions.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
plt.rcParams["font.family"] = "Times New Roman"

# Set the random seed
def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fix_seed(2025)


# Physical parameters
dh = -857142
B = 12000
E = 8556 * 8.314472
R = 8.314472
n = 3.0

# Characteristic parameters
x_max = 50e-3
t_max = 60
T_max = 1644
T_min = 300
rho_max = 280
rho_min = 220
m_max = 1e-2

# PINN parameters
n_pde = 50000
n_bc = 2500
n_ic = 5000


# Normalized function
def x_norm(x): return x / x_max


def t_norm(t): return t / t_max


def T_norm(T): return (T - T_min) / (T_max - T_min)


def rho_norm(rho): return (rho - rho_min) / (rho_max - rho_min)


def m_norm(m): return m / m_max


# Inverse normalization function
def x_inorm(x): return x * x_max


def t_inorm(t): return t * t_max


def T_inorm(T): return T_min + (T_max - T_min) * T


def rho_inorm(rho): return rho_min + (rho_max - rho_min) * rho


def m_inorm(m): return m * m_max


# Physics-informed neural networks
class PINN(nn.Module):
    def __init__(self, num_layers=6, hidden_units=60):
        super().__init__()
        layers = []
        input_units = 2
        for _ in range(num_layers):
            layers.append(nn.Linear(input_units, hidden_units))
            layers.append(nn.Tanh())
            input_units = hidden_units
        layers.append(nn.Linear(hidden_units, 1))
        layers.append(nn.Softplus())
        self.fc = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, t):
        return self.fc(torch.cat([x, t], dim=1))


# Load pretrained model
def load_models(device):
    net = PINN().to(device)
    net.load_state_dict(torch.load('net_T.pth', map_location=device))
    net.eval()
    return net


# Generate grid points
def generate_grid(nx=100, nt=100):
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    xx, tt = torch.meshgrid(x, t, indexing='ij')
    return xx, tt


# Predict and reverse-normalize
def predict_and_denormalize(net_T, xx, tt, device):
    with torch.no_grad():
        x_flat = xx.reshape(-1, 1).to(device)
        t_flat = tt.reshape(-1, 1).to(device)
        T_pred = net_T(x_flat, t_flat)
        T_actual = T_inorm(T_pred).cpu().numpy()
        return T_actual.reshape(xx.shape)


# Plot a 3D temperature cloud map
def plot_temperature_3d(xx, tt, T_actual):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_actual = x_inorm(xx)
    t_actual = t_inorm(tt)

    surf = ax.plot_surface(x_actual, t_actual, T_actual,
                           cmap=cm.jet, linewidth=0, antialiased=True)

    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Time t (s)', fontsize=12)
    ax.set_zlabel('Temperature T (K)', fontsize=12)
    ax.set_title('Temperature Field Prediction', fontsize=14)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.view_init(30, 45)

    plt.tight_layout()
    plt.savefig('temperature_field_3d.png', dpi=300)
    plt.show()


# Plot a 2D temperature contour map
def plot_temperature_contour(xx, tt, T_actual):
    plt.figure(figsize=(8, 6))

    x_actual = x_inorm(xx)
    t_actual = t_inorm(tt)

    levels = np.linspace(T_actual.min(), T_actual.max(), 20)
    cs = plt.contourf(x_actual, t_actual, T_actual, levels=levels, cmap='seismic')

    plt.xlabel('Position x (m)', fontsize=12)
    plt.ylabel('Time t (s)', fontsize=12)
    plt.title('Temperature Field Contour', fontsize=14)

    cbar = plt.colorbar(cs)
    cbar.set_label('Temperature (K)', fontsize=12)

    plt.tight_layout()
    plt.savefig('temperature_field_contour.png', dpi=300)
    plt.show()


# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    net_T = load_models(device)
    xx, tt = generate_grid(nx=200, nt=200)
    T_actual = predict_and_denormalize(net_T, xx, tt, device)
    plot_temperature_3d(xx, tt, T_actual)
    plot_temperature_contour(xx, tt, T_actual)

if __name__ == "__main__":
    main()