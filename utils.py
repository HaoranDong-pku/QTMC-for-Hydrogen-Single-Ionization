#!/usr/bin/env python3
"""
Utility functions for the QTMC simulation.
"""

import numpy as np
from scipy import constants
import os
import yaml
import argparse
from typing import Dict, Any, Optional

# Atomic units
atomic_units = {
    "length": constants.physical_constants["Bohr radius"][0],  # meters
    "energy": constants.physical_constants["Hartree energy"][0],  # joules
    "time": constants.hbar / constants.physical_constants["Hartree energy"][0],  # seconds
    "electric_field": constants.e / (4 * np.pi * constants.epsilon_0 * constants.physical_constants["Bohr radius"][0]**2)  # V/m
}

def convert_to_atomic_units(value, unit_type):
    """
    Convert a value from SI units to atomic units.
    
    Parameters:
    -----------
    value : float
        Value in SI units
    unit_type : str
        Type of unit to convert ('length', 'energy', 'time', 'electric_field')
        
    Returns:
    --------
    float
        Value in atomic units
    """
    if unit_type not in atomic_units:
        raise ValueError(f"Unknown unit type: {unit_type}")
    
    return value / atomic_units[unit_type]

def convert_from_atomic_units(value, unit_type):
    """
    Convert a value from atomic units to SI units.
    
    Parameters:
    -----------
    value : float
        Value in atomic units
    unit_type : str
        Type of unit to convert ('length', 'energy', 'time', 'electric_field')
        
    Returns:
    --------
    float
        Value in SI units
    """
    if unit_type not in atomic_units:
        raise ValueError(f"Unknown unit type: {unit_type}")
    
    return value * atomic_units[unit_type]

def gaussian_wavepacket(x, x0, p0, sigma):
    """
    Gaussian wavepacket in position space.
    
    Parameters:
    -----------
    x : ndarray
        Position grid
    x0 : float
        Central position
    p0 : float
        Central momentum
    sigma : float
        Width parameter
        
    Returns:
    --------
    ndarray
        Complex wavefunction values
    """
    norm = (2 * np.pi * sigma**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * x)

def coulomb_potential(r):
    """
    Coulomb potential for hydrogen atom in atomic units.
    
    Parameters:
    -----------
    r : float or ndarray
        Distance from the nucleus
        
    Returns:
    --------
    float or ndarray
        Coulomb potential value(s)
    """
    # V(r) = -1/r in atomic units
    return -1.0 / np.maximum(r, 1e-10)  # Avoid division by zero

def pulse_envelope(t, t0, tau, shape='gaussian'):
    """
    Calculate pulse envelope based on the selected shape.
    
    Parameters:
    -----------
    t : float
        Time in atomic units
    t0 : float
        Central time of the pulse in atomic units
    tau : float
        Pulse duration in atomic units
    shape : str
        Envelope shape: 'gaussian', 'sin2', 'trapezoidal', or 'flattop'
    
    Returns:
    --------
    float
        Envelope value at time t
    """
    if shape == 'gaussian':
        # Gaussian envelope
        return np.exp(-2.0 * np.log(2.0) * (t - t0)**2 / tau**2)
    
    elif shape == 'sin2':
        # sin² envelope
        if t < t0 - tau or t > t0 + tau:
            return 0.0
        else:
            return np.sin(np.pi * (t - (t0 - tau)) / (2 * tau))**2
    
    elif shape == 'trapezoidal':
        # Trapezoidal envelope with 20% ramp up/down
        ramp = 0.2 * tau
        if t < t0 - tau:
            return 0.0
        elif t < t0 - tau + ramp:
            return (t - (t0 - tau)) / ramp
        elif t < t0 + tau - ramp:
            return 1.0
        elif t < t0 + tau:
            return 1.0 - (t - (t0 + tau - ramp)) / ramp
        else:
            return 0.0
    
    elif shape == 'flattop':
        # Flat-top Gaussian (super-Gaussian)
        order = 10  # higher order = steeper edges
        return np.exp(-2.0 * np.log(2.0) * ((t - t0)/tau)**(2*order))
    
    else:
        raise ValueError(f"Unknown pulse shape: {shape}")

def laser_field(t, E0, omega, tau, cep=0.0, ellipticity=0.0, 
                polarization_angle=0.0, pulse_shape='gaussian'):
    """
    Laser electric field with configurable polarization and pulse shape.
    
    Parameters:
    -----------
    t : float
        Time in atomic units
    E0 : float
        Peak electric field amplitude in atomic units
    omega : float
        Laser frequency in atomic units
    tau : float
        Pulse duration in atomic units
    cep : float, optional
        Carrier-envelope phase in radians
    ellipticity : float, optional
        Ellipticity parameter (-1 to 1):
        0 = linear, 1 = right circular, -1 = left circular
    polarization_angle : float, optional
        Angle of the main polarization axis in radians
    pulse_shape : str, optional
        Shape of the pulse envelope: 'gaussian', 'sin2', 'trapezoidal', 'flattop'
    
    Returns:
    --------
    tuple
        (Ex, Ey, Ez) electric field components in atomic units
    """
    # Central time of the pulse (3x duration from start)
    t0 = 3 * tau
    
    # Calculate envelope
    envelope = pulse_envelope(t, t0, tau, shape=pulse_shape)
    
    # Carrier waves with CEP
    # For elliptical polarization, we use sin for x and cos for y component
    # with a phase determined by the ellipticity
    carrier_x = np.sin(omega * t + cep)
    
    # For elliptical polarization, the y component has a phase shift
    # Ellipticity determines the amplitude ratio between x and y components
    phase_shift = np.pi / 2  # 90 degrees for circular polarization
    carrier_y = np.sin(omega * t + cep + np.sign(ellipticity) * phase_shift)
    
    # Amplitude factors for x and y components based on ellipticity
    # For linear (ellipticity=0): Ax=1, Ay=0
    # For circular (ellipticity=±1): Ax=Ay=1/√2
    if abs(ellipticity) < 1e-10:
        # Linear polarization
        amp_x = 1.0
        amp_y = 0.0
    else:
        # Elliptical polarization
        # Scale to ensure total intensity is preserved
        amp_x = 1.0 / np.sqrt(1 + ellipticity**2)
        amp_y = abs(ellipticity) / np.sqrt(1 + ellipticity**2)
    
    # Calculate field components in the pulse's natural frame
    Ex_natural = E0 * envelope * amp_x * carrier_x
    Ey_natural = E0 * envelope * amp_y * carrier_y
    
    # Rotate to the desired polarization angle
    cos_angle = np.cos(polarization_angle)
    sin_angle = np.sin(polarization_angle)
    
    Ex = Ex_natural * cos_angle - Ey_natural * sin_angle
    Ey = Ex_natural * sin_angle + Ey_natural * cos_angle
    Ez = 0.0  # We're keeping the field in the xy plane
    
    return (Ex, Ey, Ez)

def adk_ionization_rate(E, Ip=0.5):
    """
    Simplified ADK tunneling ionization rate for hydrogen atom.
    
    Parameters:
    -----------
    E : float
        Electric field strength in atomic units
    Ip : float, optional
        Ionization potential in atomic units (0.5 for H atom)
    
    Returns:
    --------
    float
        Ionization rate in atomic units
    """
    if E < 1e-10:
        return 0.0
    
    # Constants for hydrogen
    n_eff = 1.0
    l = 0
    m = 0
    
    # Simplified ADK formula
    prefactor = (3 * n_eff**3 * E) / (np.pi * np.sqrt(2 * Ip))
    exp_term = -2 * n_eff**3 / (3 * E)
    
    return prefactor * np.exp(exp_term)

def momentum_to_energy(p):
    """
    Convert momentum to energy in atomic units.
    
    Parameters:
    -----------
    p : ndarray
        Momentum vector [px, py, pz] in atomic units
        
    Returns:
    --------
    float
        Energy in atomic units
    """
    return 0.5 * np.sum(p**2)

def plot_laser_field(E0, omega, tau, cep=0.0, ellipticity=0.0, 
                      polarization_angle=0.0, pulse_shape='gaussian', 
                      t_max=None, num_points=1000, save_path=None):
    """
    Plot the laser electric field for visualization.
    
    Parameters:
    -----------
    E0 : float
        Peak electric field amplitude in atomic units
    omega : float
        Laser frequency in atomic units
    tau : float
        Pulse duration in atomic units
    cep : float, optional
        Carrier-envelope phase in radians
    ellipticity : float, optional
        Ellipticity parameter (0=linear, ±1=circular)
    polarization_angle : float, optional
        Angle of the main polarization axis in radians
    pulse_shape : str, optional
        Shape of the pulse envelope
    t_max : float, optional
        Maximum time to plot (default: 6*tau)
    num_points : int, optional
        Number of points to plot
    save_path : str, optional
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    if t_max is None:
        t_max = 6 * tau  # Plot 6 times the pulse duration
    
    # Create time array
    times = np.linspace(0, t_max, num_points)
    
    # Calculate field at each time
    Ex = []
    Ey = []
    for t in times:
        ex, ey, _ = laser_field(t, E0, omega, tau, cep, ellipticity, 
                                 polarization_angle, pulse_shape)
        Ex.append(ex)
        Ey.append(ey)
    
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    E_total = np.sqrt(Ex**2 + Ey**2)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # First subplot: Electric field components vs time
    ax1 = fig.add_subplot(211)
    ax1.plot(times, Ex, 'b-', label='Ex')
    ax1.plot(times, Ey, 'r-', label='Ey')
    ax1.plot(times, E_total, 'k--', label='|E|', alpha=0.5)
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Electric Field (a.u.)')
    ax1.set_title(f'Laser Pulse: {pulse_shape.capitalize()}, ε={ellipticity}, φ={cep}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Second subplot: Polarization ellipse (Ex vs Ey)
    ax2 = fig.add_subplot(212)
    ax2.plot(Ex, Ey, 'g-', alpha=0.7)
    ax2.set_xlabel('Ex (a.u.)')
    ax2.set_ylabel('Ey (a.u.)')
    ax2.set_title('Polarization Ellipse')
    
    # Make the plot square with equal aspect ratio
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

###############################################################################
# Configuration Management
###############################################################################

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a configuration file from the configs directory.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration file (without extension)
        
    Returns:
    --------
    dict
        Dictionary containing the configuration
    """
    # Create configs directory if it doesn't exist
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    
    # Try to load from configs directory
    config_path = os.path.join(configs_dir, f"{config_name}.yaml")
    
    # If not found, check if it's an absolute path
    if not os.path.exists(config_path) and os.path.exists(config_name):
        config_path = config_name
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_name: str) -> str:
    """
    Save a configuration to a YAML file.
    
    Parameters:
    -----------
    config : dict
        Dictionary containing the configuration
    config_name : str
        Name of the configuration file (without extension)
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    # Create configs directory if it doesn't exist
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    
    config_path = os.path.join(configs_dir, f"{config_name}.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def list_configs() -> list:
    """
    List all available configuration files.
    
    Returns:
    --------
    list
        List of configuration file names (without extension)
    """
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
        return []
    
    configs = []
    for file in os.listdir(configs_dir):
        if file.endswith('.yaml'):
            configs.append(file[:-5])  # Remove .yaml extension
    
    return configs

def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert command-line arguments to a configuration dictionary.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    dict
        Dictionary containing the configuration
    """
    config = vars(args).copy()  # Convert namespace to dict
    
    # Organize parameters into categories
    result = {
        "simulation": {
            "trajectories": config.get("trajectories", 1000),
            "t_max": config.get("t_max", 1000),
            "dt": config.get("dt", 0.1),
        },
        "laser": {
            "intensity": config.get("intensity", 1.0e14),
            "wavelength": config.get("wavelength", 800),
            "pulse_duration": config.get("pulse_duration", 10.0),
            "cep": config.get("cep", 0.0),
            "ellipticity": config.get("ellipticity", 0.0),
            "polarization_angle": config.get("polarization_angle", 0.0),
            "pulse_shape": config.get("pulse_shape", "gaussian"),
        },
        "output": {
            "output_dir": config.get("output_dir", "results"),
            "save_plots": config.get("save_plots", False),
            "plot_laser": config.get("plot_laser", False),
        },
        "visualization": {
            "quantum_spectrum": config.get("quantum_spectrum", False),
            "grid_size": config.get("grid_size", 200),
            "p_range": config.get("p_range", 1.5),
            "log_scale": config.get("log_scale", False),
        },
        "parallel": {
            "serial": config.get("serial", False),
            "processes": config.get("processes", None),
        }
    }
    
    return result

def config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a configuration dictionary to a flat dictionary suitable for command-line arguments.
    
    Parameters:
    -----------
    config : dict
        Dictionary containing the structured configuration
        
    Returns:
    --------
    dict
        Flat dictionary containing command-line arguments
    """
    args = {}
    
    # Flatten nested dictionaries
    for category, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                args[key] = value
        else:
            args[category] = params
    
    return args

def create_example_configs():
    """
    Create example configuration files if they don't exist.
    """
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    
    # Example 1: Linear polarization with high resolution
    linear_config = {
        "simulation": {
            "trajectories": 5000,
            "t_max": 1000,
            "dt": 0.1,
        },
        "laser": {
            "intensity": 1.0e14,
            "wavelength": 800,
            "pulse_duration": 10.0,
            "cep": 0.0,
            "ellipticity": 0.0,
            "polarization_angle": 0.0,
            "pulse_shape": "sin2",
        },
        "output": {
            "output_dir": "results/linear",
            "save_plots": True,
            "plot_laser": True,
        },
        "visualization": {
            "quantum_spectrum": True,
            "grid_size": 300,
            "p_range": 1.5,
            "log_scale": True,
        },
        "parallel": {
            "serial": False,
            "processes": None,
        }
    }
    
    # Example 2: Circular polarization
    circular_config = {
        "simulation": {
            "trajectories": 5000,
            "t_max": 1000,
            "dt": 0.1,
        },
        "laser": {
            "intensity": 1.0e14,
            "wavelength": 800,
            "pulse_duration": 10.0,
            "cep": 0.0,
            "ellipticity": 1.0,
            "polarization_angle": 0.0,
            "pulse_shape": "sin2",
        },
        "output": {
            "output_dir": "results/circular",
            "save_plots": True,
            "plot_laser": True,
        },
        "visualization": {
            "quantum_spectrum": True,
            "grid_size": 300,
            "p_range": 1.5,
            "log_scale": True,
        },
        "parallel": {
            "serial": False,
            "processes": None,
        }
    }
    
    # Example 3: Fast test run (fewer trajectories)
    test_config = {
        "simulation": {
            "trajectories": 100,
            "t_max": 500,
            "dt": 0.2,
        },
        "laser": {
            "intensity": 1.0e14,
            "wavelength": 800,
            "pulse_duration": 5.0,
            "cep": 0.0,
            "ellipticity": 0.0,
            "polarization_angle": 0.0,
            "pulse_shape": "gaussian",
        },
        "output": {
            "output_dir": "results/test",
            "save_plots": True,
            "plot_laser": False,
        },
        "visualization": {
            "quantum_spectrum": False,
            "grid_size": 100,
            "p_range": 1.5,
            "log_scale": True,
        },
        "parallel": {
            "serial": False,
            "processes": None,
        }
    }
    
    # Example 4: Elliptical polarization at 45 degrees
    elliptical_config = {
        "simulation": {
            "trajectories": 5000,
            "t_max": 1000,
            "dt": 0.1,
        },
        "laser": {
            "intensity": 1.0e14,
            "wavelength": 800,
            "pulse_duration": 10.0,
            "cep": 0.0,
            "ellipticity": 0.5,
            "polarization_angle": 0.785,  # 45 degrees in radians
            "pulse_shape": "flattop",
        },
        "output": {
            "output_dir": "results/elliptical",
            "save_plots": True,
            "plot_laser": True,
        },
        "visualization": {
            "quantum_spectrum": True,
            "grid_size": 300,
            "p_range": 1.5,
            "log_scale": True,
        },
        "parallel": {
            "serial": False,
            "processes": None,
        }
    }
    
    # Save example configurations
    configs = {
        "linear": linear_config,
        "circular": circular_config,
        "test": test_config,
        "elliptical": elliptical_config
    }
    
    for name, config in configs.items():
        config_path = os.path.join(configs_dir, f"{name}.yaml")
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Created example configuration: {config_path}") 