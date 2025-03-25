# QTMC Simulation for Hydrogen Atom Ionization

A Quantum Trajectory Monte Carlo (QTMC) simulation for single electron ionization of a hydrogen atom under a strong laser field.

## Overview

This project implements a QTMC approach to simulate the ionization process of a hydrogen atom in a strong laser field. The simulation combines quantum mechanics with classical trajectory calculations to model the electron dynamics after tunneling ionization.

Key physics concepts modeled:
- Tunneling ionization of the electron
- ADK (Ammosov-Delone-Krainov) ionization rates
- Coulomb potential of the atomic core
- Laser-electron interaction with various polarization states
- Electron trajectories post-ionization

## Features

- Full 3D electron trajectory calculations
- Advanced laser field configuration:
  - Elliptical and circular polarization support
  - Multiple pulse envelope shapes (Gaussian, sin², trapezoidal, flat-top)
  - Configurable carrier-envelope phase (CEP)
  - Adjustable polarization angle
- Parallel computing support for faster simulations
- Comprehensive visualization tools
- Configurable simulation parameters
- Command-line interface for easy use
- Configuration file system for parameter management
- Quantum momentum spectrum with interference effects

## Files

- `qtmc.py`: Main simulation class implementing the QTMC approach
- `utils.py`: Utility functions, physical constants, and helper methods
- `visualize.py`: Visualization tools for analyzing simulation results
- `main.py`: CLI script for running simulations with different parameters
- `requirements.txt`: Required Python packages
- `configs/`: Directory with saved configuration files (YAML)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/qtmc-hydrogen.git
cd qtmc-hydrogen
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run a basic simulation with default parameters:

```
python main.py
```

Customize your simulation with command-line arguments:

```
python main.py --trajectories 5000 --intensity 5.0e14 --wavelength 800 --pulse-duration 10.0 --save-plots
```

### Configuration System

Instead of typing numerous command-line parameters each time, you can use configuration files to save and load parameter sets:

```
# Create example configuration files
python main.py --create-examples

# List available configurations
python main.py --list-configs

# Run a simulation using a configuration file
python main.py --config linear

# Override specific parameters from a configuration
python main.py --config linear --trajectories 2000 --ellipticity 0.2

# Save current parameters to a configuration file
python main.py --trajectories 5000 --quantum-spectrum --save-config my_config
```

Configuration files are stored as human-readable YAML files in the `configs/` directory with parameters organized into logical categories:

```yaml
simulation:
  trajectories: 5000
  t_max: 1000
  dt: 0.1
laser:
  intensity: 1.0e+14
  wavelength: 800
  pulse_duration: 10.0
  ellipticity: 0.0
  # ...
```

The pre-defined example configurations include:
- `linear`: Linear polarization with high resolution
- `circular`: Circular polarization
- `elliptical`: Elliptical polarization at 45°
- `test`: Fast test run with fewer trajectories

### Laser Field Configuration

The simulation supports various laser field configurations:

```
# Linear polarization (default)
python main.py --ellipticity 0 --pulse-shape gaussian

# Circular polarization
python main.py --ellipticity 1 --pulse-shape sin2

# Elliptical polarization at 45 degrees
python main.py --ellipticity 0.5 --polarization-angle 0.785 --pulse-shape flattop

# Preview laser field without running simulation
python main.py --plot-laser --trajectories 0
```

### Parallel Computing

By default, the simulation uses all available CPU cores for parallel computation of trajectories. You can control this behavior with:

```
# Disable parallel computing
python main.py --serial

# Specify number of processes
python main.py --processes 4
```

For large numbers of trajectories, parallel computing can provide significant speedups:

| Number of Trajectories | Serial Time | Parallel Time (8 cores) | Speedup |
|------------------------|-------------|-------------------------|---------|
| 100                    | ~1 min      | ~15 sec                 | ~4x     |
| 1000                   | ~10 min     | ~1.5 min                | ~6.7x   |
| 10000                  | ~100 min    | ~14 min                 | ~7.1x   |

*Note: Performance may vary based on your hardware configuration*

### Available Options

#### Configuration Options
- `--config`: Load parameters from a configuration file (name or path)
- `--save-config`: Save current parameters to a configuration file
- `--list-configs`: List available configuration files
- `--create-examples`: Create example configuration files

#### Simulation Parameters
- `--trajectories`: Number of Monte Carlo trajectories (default: 1000)
- `--t-max`: Maximum simulation time in atomic units (default: 1000)
- `--dt`: Time step in atomic units (default: 0.1)

#### Laser Parameters
- `--intensity`: Laser intensity in W/cm² (default: 1.0e14)
- `--wavelength`: Laser wavelength in nm (default: 800)
- `--pulse-duration`: Pulse duration in fs (default: 10.0)
- `--cep`: Carrier-envelope phase in radians (default: 0.0)
- `--ellipticity`: Ellipticity parameter: 0=linear, 1=right circular, -1=left circular (default: 0.0)
- `--polarization-angle`: Polarization angle in radians (default: 0.0)
- `--pulse-shape`: Pulse envelope shape: 'gaussian', 'sin2', 'trapezoidal', 'flattop' (default: 'gaussian')

#### Output Parameters
- `--output-dir`: Directory to save results (default: results)
- `--save-plots`: Save plots and data to output directory
- `--plot-laser`: Plot the laser field before running the simulation

#### Visualization Parameters
- `--quantum-spectrum`: Generate quantum momentum spectrum with interference effects
- `--grid-size`: Grid size for quantum momentum spectrum (default: 200)
- `--p-range`: Momentum range for visualization (default: 1.5 a.u.)
- `--log-scale`: Use logarithmic scale for spectral plots

#### Parallel Computing Parameters
- `--serial`: Disable parallel computing
- `--processes`: Number of processes to use (default: all available cores)

## Physics Background

### Quantum Trajectory Monte Carlo (QTMC)

QTMC is a semiclassical approach that combines quantum mechanical ionization with classical propagation of electron trajectories. The process involves:

1. **Tunneling Ionization**: Quantum mechanical tunneling through the potential barrier, calculated using ADK theory
2. **Initial Conditions**: The electron emerges with initial conditions from the tunnel exit
3. **Classical Propagation**: Classical equations of motion are solved to propagate the electron in the combined laser and Coulomb fields
4. **Statistical Analysis**: Many trajectories are calculated to build up momentum, energy, and angular distributions

### Laser Polarization and Pulse Shapes

The simulation supports different polarization states:
- **Linear polarization** (ellipticity = 0): Electric field oscillates along a single axis
- **Circular polarization** (ellipticity = 1 or -1): Electric field rotates in a circle, with constant amplitude
- **Elliptical polarization** (0 < |ellipticity| < 1): Electric field traces an ellipse

Different pulse shapes affect how the laser field turns on and off:
- **Gaussian**: Smooth Gaussian envelope (standard)
- **sin²**: Smoother edges with a more flat-top central region
- **Trapezoidal**: Flat central region with linear ramps at the edges
- **Flat-top**: Steep edges with an extended flat plateau (super-Gaussian)

### Atomic Units

The simulation uses atomic units where:
- Electron mass (mₑ) = 1
- Elementary charge (e) = 1
- Reduced Planck's constant (ħ) = 1
- Coulomb constant (kₑ = 1/4πε₀) = 1

In these units:
- The Bohr radius a₀ = 1
- The energy unit is the Hartree = 27.2 eV
- The time unit is 2.42 × 10⁻¹⁷ s

## Visualizations

The simulation generates several visualizations:

- Laser field visualization (time profile and polarization ellipse)
- 2D momentum distributions in px-pz, px-py, and py-pz planes
- 3D momentum distribution
- Energy spectrum of ionized electrons
- Angular distribution of ionized electrons
- Ionization time vs. laser field
- Sample electron trajectories in polarization-relevant planes
- Animated electron trajectories (if supported)

## Performance Considerations

- The simulation is computationally intensive, especially for large numbers of trajectories
- Parallel computing significantly reduces computation time
- For very large simulations (>10,000 trajectories), consider running on a high-performance computing system
- Visualization of 3D momentum distributions can be memory-intensive
- Elliptical polarization simulations require more complex calculations but provide richer physics insights

## Contributing

Contributions to improve the simulation are welcome. Areas for potential improvement:

- More sophisticated ADK model for ionization
- Inclusion of non-adiabatic effects
- Further optimization of parallel computing implementation
- GPU acceleration for trajectory calculations
- Addition of more realistic pulse shapes
- Incorporation of multi-electron effects
- Implementation of more advanced visualization techniques for 3D trajectories

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Ammosov, M. V., Delone, N. B., & Krainov, V. P. (1986). Tunnel ionization of complex atoms and of atomic ions in an alternating electromagnetic field. Sov. Phys. JETP, 64(6), 1191-1194.
2. Brabec, T., Ivanov, M. Y., & Corkum, P. B. (1996). Coulomb focusing in intense field atomic processes. Physical Review A, 54(4), R2551.
3. Lewenstein, M., Balcou, P., Ivanov, M. Y., L'Huillier, A., & Corkum, P. B. (1994). Theory of high-harmonic generation by low-frequency laser fields. Physical Review A, 49(3), 2117.
4. Milošević, D. B., Paulus, G. G., Bauer, D., & Becker, W. (2006). Above-threshold ionization by few-cycle pulses. Journal of Physics B: Atomic, Molecular and Optical Physics, 39(14), R203. 