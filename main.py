#!/usr/bin/env python3
"""
Main script for running QTMC simulations of hydrogen atom ionization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import multiprocessing as mp
import yaml
from typing import Dict, Any

from qtmc import QTMCSimulation
from utils import (
    laser_field, 
    load_config, 
    save_config, 
    list_configs, 
    args_to_config, 
    config_to_args,
    create_example_configs
)
from visualize import (
    plot_momentum_spectrum_2d,
    plot_momentum_spectrum_3d,
    plot_energy_spectrum,
    plot_angular_distribution,
    plot_ionization_times,
    plot_trajectory_animation,
    plot_quantum_momentum_spectrum
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QTMC simulation for H atom ionization")
    
    # Configuration file options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config', type=str, 
                        help='Load parameters from a configuration file (name or path)')
    config_group.add_argument('--save-config', type=str, 
                        help='Save current parameters to a configuration file')
    config_group.add_argument('--list-configs', action='store_true',
                        help='List available configuration files')
    config_group.add_argument('--create-examples', action='store_true',
                        help='Create example configuration files')
    
    # Simulation parameters
    parser.add_argument('--trajectories', type=int, default=1000,
                        help='Number of Monte Carlo trajectories (default: 1000)')
    parser.add_argument('--t-max', type=float, default=1000,
                        help='Maximum simulation time in atomic units (default: 1000)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step in atomic units (default: 0.1)')
    
    # Laser parameters
    laser_group = parser.add_argument_group('Laser Parameters')
    laser_group.add_argument('--intensity', type=float, default=1.0e14,
                        help='Laser intensity in W/cm^2 (default: 1.0e14)')
    laser_group.add_argument('--wavelength', type=float, default=800,
                        help='Laser wavelength in nm (default: 800)')
    laser_group.add_argument('--pulse-duration', type=float, default=10.0,
                        help='Pulse duration in fs (default: 10.0)')
    laser_group.add_argument('--cep', type=float, default=0.0,
                        help='Carrier-envelope phase in radians (default: 0.0)')
    laser_group.add_argument('--ellipticity', type=float, default=0.0,
                        help='Ellipticity parameter: 0=linear, 1=right circular, -1=left circular (default: 0.0)')
    laser_group.add_argument('--polarization-angle', type=float, default=0.0,
                        help='Polarization angle in radians (default: 0.0)')
    laser_group.add_argument('--pulse-shape', type=str, default='gaussian', 
                        choices=['gaussian', 'sin2', 'trapezoidal', 'flattop'],
                        help='Pulse envelope shape (default: gaussian)')
    
    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    output_group.add_argument('--save-plots', action='store_true',
                        help='Save plots to output directory')
    output_group.add_argument('--plot-laser', action='store_true',
                        help='Plot the laser field before running the simulation')
    
    # Visualization parameters
    vis_group = parser.add_argument_group('Visualization Parameters')
    vis_group.add_argument('--quantum-spectrum', action='store_true',
                        help='Generate quantum momentum spectrum with interference effects')
    vis_group.add_argument('--grid-size', type=int, default=200,
                        help='Grid size for quantum momentum spectrum (default: 200)')
    vis_group.add_argument('--p-range', type=float, default=1.5,
                        help='Momentum range for visualization (+/- this value, default: 1.5 a.u.)')
    vis_group.add_argument('--log-scale', action='store_true',
                        help='Use logarithmic scale for spectral plots')
    
    # Parallel computing parameters
    parallel_group = parser.add_argument_group('Parallel Computing')
    parallel_group.add_argument('--serial', action='store_true',
                        help='Run in serial mode (disable parallel computing)')
    parallel_group.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use for parallel computing (default: all available cores)')
    
    return parser, parser.parse_args()

def main():
    """Run the QTMC simulation with the specified parameters."""
    parser, args = parse_args()
    
    # Handle configuration file options
    if args.list_configs:
        configs = list_configs()
        if configs:
            print("Available configuration files:")
            for config in configs:
                print(f"  - {config}")
        else:
            print("No configuration files found.")
        return
    
    if args.create_examples:
        create_example_configs()
        print("Example configuration files created.")
        return
    
    # Load configuration file if specified
    if args.config:
        try:
            print(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
            # Convert structured config to flat args dict
            config_args = config_to_args(config)
            
            # Create a new namespace with default values
            new_args = argparse.Namespace()
            for arg_name, arg_value in vars(args).items():
                # If the argument is in the config and not explicitly set on command line,
                # use the config value. Otherwise, use the command line value.
                if arg_name in config_args and arg_value == parser.get_default(arg_name):
                    setattr(new_args, arg_name, config_args[arg_name])
                else:
                    setattr(new_args, arg_name, arg_value)
            
            args = new_args
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return
    
    # Save current configuration if specified
    if args.save_config:
        try:
            config = args_to_config(args)
            config_path = save_config(config, args.save_config)
            print(f"Configuration saved to: {config_path}")
            
            # If this is the only action requested, return
            if not args.trajectories:
                return
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return
    
    # Create output directory if it doesn't exist
    if args.save_plots and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get number of available cores for display
    available_cores = mp.cpu_count()
    num_processes = args.processes if args.processes is not None else available_cores
    
    # Create a simulation instance
    simulation = QTMCSimulation(
        num_trajectories=args.trajectories,
        t_max=args.t_max,
        dt=args.dt,
        laser_intensity=args.intensity,
        laser_wavelength=args.wavelength,
        pulse_duration=args.pulse_duration,
        cep=args.cep,
        ellipticity=args.ellipticity,
        polarization_angle=args.polarization_angle,
        pulse_shape=args.pulse_shape,
        n_processes=num_processes
    )
    
    # Print simulation parameters
    print("\nSimulation Parameters:")
    print(f"  Trajectories: {args.trajectories}")
    print(f"  Laser: {args.intensity:.1e} W/cm², {args.wavelength} nm, {args.pulse_duration} fs")
    print(f"  Polarization: Ellipticity={args.ellipticity}, Angle={args.polarization_angle:.2f} rad")
    print(f"  Pulse Shape: {args.pulse_shape}")
    
    # Print parallel computing info
    if not args.serial:
        print(f"\nParallel computing enabled with {num_processes} out of {available_cores} available CPU cores")
    else:
        print("\nParallel computing disabled, running in serial mode")
    
    # Create plotting function
    def save_path(filename):
        return os.path.join(args.output_dir, filename) if args.save_plots else None
    
    # Plot laser field if requested
    if args.plot_laser:
        print("Plotting laser field...")
        simulation.plot_laser_field(save_path=save_path('laser_field.png'))
    
    # Run the simulation
    simulation.run_simulation(parallel=not args.serial)
    
    # Get results
    momenta = simulation.momentum_distribution()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Quantum momentum spectrum with interference effects - new feature
    if args.quantum_spectrum and hasattr(simulation, 'quantum_phases'):
        print("Generating quantum momentum spectrum with interference effects...")
        # Calculate momentum range based on argument
        p_range = (-args.p_range, args.p_range)
        
        # Calculate energy parameters for ATI rings
        ponderomotive_energy = simulation.E0**2 / (4 * simulation.omega**2)
        photon_energy = simulation.omega
        ionization_potential = simulation.Ip
        
        # Generate quantum spectra for relevant planes
        # Primary polarization plane (px-pz for linear polarization)
        plot_quantum_momentum_spectrum(
            momenta, simulation.quantum_phases, 
            plane='pxpz',
            p_range=p_range,
            grid_size=args.grid_size,
            log_scale=args.log_scale,
            save_path=save_path('quantum_momentum_pxpz.png'),
            ponderomotive_energy=ponderomotive_energy,
            photon_energy=photon_energy,
            ionization_potential=ionization_potential
        )
        
        # For elliptical/circular polarization, also show px-py plane
        if abs(args.ellipticity) > 0.1:
            plot_quantum_momentum_spectrum(
                momenta, simulation.quantum_phases, 
                plane='pxpy',
                p_range=p_range,
                grid_size=args.grid_size,
                log_scale=args.log_scale,
                save_path=save_path('quantum_momentum_pxpy.png'),
                ponderomotive_energy=ponderomotive_energy,
                photon_energy=photon_energy,
                ionization_potential=ionization_potential
            )
    
    # Classical 2D momentum distributions for different planes
    print("Generating classical momentum spectra...")
    plot_momentum_spectrum_2d(momenta, plane='pxpz', bins=args.grid_size, 
                             log_scale=args.log_scale, save_path=save_path('momentum_pxpz.png'))
    plot_momentum_spectrum_2d(momenta, plane='pxpy', bins=args.grid_size, 
                             log_scale=args.log_scale, save_path=save_path('momentum_pxpy.png'))
    
    # For elliptical polarization, also show the pypz plane
    if abs(args.ellipticity) > 0.1:
        plot_momentum_spectrum_2d(momenta, plane='pypz', bins=args.grid_size, 
                                 log_scale=args.log_scale, save_path=save_path('momentum_pypz.png'))
    
    # 3D momentum distribution
    plot_momentum_spectrum_3d(momenta, method='scatter', markers=1000, 
                             save_path=save_path('momentum_3d.png'))
    
    # Energy spectrum
    plot_energy_spectrum(momenta, save_path=save_path('energy_spectrum.png'))
    
    # Angular distribution
    plot_angular_distribution(momenta, save_path=save_path('angular_distribution.png'))
    
    # Ionization times
    field_function = lambda t: laser_field(
        t, simulation.E0, simulation.omega, simulation.pulse_duration_au,
        simulation.cep, simulation.ellipticity, simulation.polarization_angle, 
        simulation.pulse_shape
    )
    plot_ionization_times(simulation.ionization_times, field_function, 
                         simulation.t_max, simulation.dt, 
                         save_path=save_path('ionization_times.png'))
    
    # Phase distribution if quantum phases are available
    if hasattr(simulation, 'quantum_phases') and simulation.quantum_phases:
        simulation.plot_phase_distribution(save_path=save_path('phase_distribution.png'))
    
    # Trajectory plots - show in relevant planes based on polarization
    simulation.plot_trajectories(num_to_plot=10, save_path=save_path('trajectories_xz.png'))
    
    # If using elliptical polarization, also show xy plane
    if abs(args.ellipticity) > 0.1:
        # Custom trajectory plot for xy plane
        if simulation.trajectories:
            plt.figure(figsize=(12, 10))
            
            indices = np.random.choice(len(simulation.trajectories), 
                                      min(10, len(simulation.trajectories)), 
                                      replace=False)
            
            for idx in indices:
                traj = simulation.trajectories[idx]
                plt.plot(traj.y[0], traj.y[1], linewidth=1, alpha=0.7)
                
            plt.xlabel('x (a.u.)')
            plt.ylabel('y (a.u.)')
            plt.title('Sample of 10 Electron Trajectories (x-y plane)')
            plt.grid(alpha=0.3)
            
            if args.save_plots:
                plt.savefig(os.path.join(args.output_dir, 'trajectories_xy.png'), dpi=300)
                
            plt.show()
    
    # Try to create trajectory animation if matplotlib animation is available
    try:
        anim = plot_trajectory_animation(simulation.trajectories, 
                                        save_path=save_path('trajectory_animation.gif'))
        print("Animation created and saved.")
    except Exception as e:
        print(f"Could not create animation: {e}")
    
    print("All visualizations completed.")
    
    # Save numerical results
    if args.save_plots:
        np.save(os.path.join(args.output_dir, 'momenta.npy'), momenta)
        np.save(os.path.join(args.output_dir, 'ionization_times.npy'), 
               np.array(simulation.ionization_times))
        if hasattr(simulation, 'quantum_phases') and simulation.quantum_phases:
            np.save(os.path.join(args.output_dir, 'quantum_phases.npy'),
                   np.array(simulation.quantum_phases))
        print(f"Numerical results saved to {args.output_dir}")
    
    print("Simulation complete!")
    
    # Print some example commands for different configurations
    print("\nTry these example commands:")
    print("  List available configurations:")
    print("    python main.py --list-configs")
    print("  Create example configuration files:")
    print("    python main.py --create-examples")
    print("  Load a configuration:")
    print("    python main.py --config linear")
    print("  Load a configuration and override some parameters:")
    print("    python main.py --config linear --trajectories 2000 --ellipticity 0.2")
    print("  Save current parameters to a configuration file:")
    print("    python main.py --trajectories 5000 --quantum-spectrum --save-config my_config")
    
if __name__ == "__main__":
    main() 