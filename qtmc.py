#!/usr/bin/env python3
"""
Quantum Trajectory Monte Carlo (QTMC) simulation for single electron ionization of H atom.
With parallel computing support.
"""

import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import atomic_units, gaussian_wavepacket, coulomb_potential, laser_field, plot_laser_field
import multiprocessing as mp
from functools import partial

class QTMCSimulation:
    def __init__(self, 
                 num_trajectories=1000, 
                 t_max=3000,  # in atomic units
                 dt=0.1,      # in atomic units
                 laser_intensity=1.0e14,  # W/cm^2
                 laser_wavelength=800,    # nm
                 pulse_duration=30.0,     # fs
                 cep=0.0,                 # carrier-envelope phase
                 ellipticity=0.0,         # ellipticity parameter
                 polarization_angle=0.0,  # polarization angle in radians
                 pulse_shape='gaussian',  # pulse envelope shape
                 Ip=0.5,                  # ionization potential (0.5 a.u. for H atom)
                 Z=1,                     # nuclear charge (1 for H atom)
                 n_processes=None):       # number of processes for parallel computing
        """
        Initialize QTMC simulation for hydrogen atom ionization.
        
        Parameters:
        -----------
        num_trajectories : int
            Number of Monte Carlo trajectories to simulate
        t_max : float
            Maximum simulation time in atomic units
        dt : float
            Time step in atomic units
        laser_intensity : float
            Peak laser intensity in W/cm^2
        laser_wavelength : float
            Laser wavelength in nm
        pulse_duration : float
            Pulse duration in femtoseconds
        cep : float
            Carrier-envelope phase in radians
        ellipticity : float
            Ellipticity parameter (-1 to 1):
            0 = linear, 1 = right circular, -1 = left circular
        polarization_angle : float
            Angle of the main polarization axis in radians
        pulse_shape : str
            Shape of the pulse envelope: 'gaussian', 'sin2', 'trapezoidal', 'flattop'
        Ip : float
            Ionization potential in atomic units (0.5 a.u. for H atom)
        Z : int
            Nuclear charge (1 for H atom)
        n_processes : int, optional
            Number of processes for parallel computing. If None, uses the number of available CPU cores.
        """
        self.num_trajectories = num_trajectories
        self.t_max = t_max
        self.dt = dt
        self.time_steps = int(t_max / dt)
        self.time_array = np.linspace(0, t_max, self.time_steps)
        
        # Laser parameters
        self.laser_intensity = laser_intensity
        self.laser_wavelength = laser_wavelength
        self.pulse_duration = pulse_duration
        self.cep = cep
        self.ellipticity = ellipticity
        self.polarization_angle = polarization_angle
        self.pulse_shape = pulse_shape
        
        # Atom parameters
        self.Ip = Ip
        self.Z = Z
        
        # Calculated in atomic units
        self.E0 = np.sqrt(laser_intensity / (3.51e16))  # Peak electric field
        self.omega = 2.0 * np.pi * constants.c / (laser_wavelength * 1e-9) * atomic_units["time"]
        self.pulse_duration_au = pulse_duration * 1e-15 / atomic_units["time"]
        
        # Results storage
        self.trajectories = []
        self.ionization_times = []
        self.final_momenta = []
        self.quantum_phases = []
        self.ionization_prob = 0.0
        
        # Parallel computing setup
        self.n_processes = n_processes if n_processes is not None else mp.cpu_count()
    
    def _equations_of_motion(self, t, y):
        """
        Equations of motion for the electron after ionization.
        y = [x, y, z, px, py, pz]
        """
        x, y, z, px, py, pz = y
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Laser field at time t
        Ex, Ey, Ez = laser_field(
            t, self.E0, self.omega, self.pulse_duration_au, 
            self.cep, self.ellipticity, self.polarization_angle, self.pulse_shape
        )
        
        # Coulomb force
        if r < 1e-10:  # Avoid division by zero
            Fx, Fy, Fz = 0, 0, 0
        else:
            Fx = -self.Z * x / r**3
            Fy = -self.Z * y / r**3
            Fz = -self.Z * z / r**3
        
        # Equations of motion
        dx_dt = px
        dy_dt = py
        dz_dt = pz
        dpx_dt = Fx + Ex  # Laser force along x-axis
        dpy_dt = Fy + Ey  # Laser force along y-axis
        dpz_dt = Fz + Ez  # Laser force along z-axis
        
        return [dx_dt, dy_dt, dz_dt, dpx_dt, dpy_dt, dpz_dt]
    
    def _calculate_quantum_phase(self, sol, t_ion, y0):
        """
        Calculate the quantum phase for a trajectory.
        
        Φ = -v₀⋅r₀ + Iₚt₀ - ∫[v(t)²/2 - 2Z/r(t)]dt
        
        Parameters:
        -----------
        sol : OdeSolution
            Solution of the equations of motion
        t_ion : float
            Ionization time
        y0 : ndarray
            Initial conditions [x0, y0, z0, px0, py0, pz0]
            
        Returns:
        --------
        float
            Quantum phase in radians
        """
        try:
            # Extract initial position and velocity
            r0 = y0[0:3]
            v0 = y0[3:6]
            
            # Calculate the first term: -v₀⋅r₀
            term1 = -np.dot(v0, r0)
            
            # Calculate the second term: Iₚt₀
            term2 = self.Ip * t_ion
            
            # Calculate the integral term
            # Extract position and velocity over time
            times = sol.t
            positions = sol.y[0:3, :]
            velocities = sol.y[3:6, :]
            
            # Calculate r(t) = |r(t)|
            r_values = np.sqrt(np.sum(positions**2, axis=0))
            
            # Calculate v(t)² = |v(t)|²
            v_squared_values = np.sum(velocities**2, axis=0)
            
            # Calculate the integrand: v(t)²/2 - 2Z/r(t)
            # Make sure to handle possible division by zero
            r_safe = np.maximum(r_values, 1e-10)
            integrand = v_squared_values/2 - 2*self.Z/r_safe
            
            # Perform numerical integration using Simpson's rule
            term3 = -simpson(integrand, times)
            
            # Sum all terms to get the quantum phase
            phase = term1 + term2 + term3
            
            return phase
            
        except Exception as e:
            print(f"Error calculating quantum phase: {str(e)}")
            return 0.0
    
    def _generate_ionization_time(self):
        """
        Generate an ionization time based on the ADK ionization rate.
        Returns the ionization time in atomic units.
        """
        # Simplified ADK model - in a real simulation this would be more sophisticated
        times = self.time_array
        
        # Collect field strengths
        field_amplitudes = []
        for t in times:
            Ex, Ey, Ez = laser_field(
                t, self.E0, self.omega, self.pulse_duration_au,
                self.cep, self.ellipticity, self.polarization_angle, self.pulse_shape
            )
            # Total field amplitude
            field_amp = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            field_amplitudes.append(field_amp)
        
        field_amplitudes = np.array(field_amplitudes)
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        safe_field_amplitudes = np.maximum(field_amplitudes, epsilon)
        
        # ADK rate (simplified)
        ionization_rates = np.exp(-2.0 / (3.0 * safe_field_amplitudes)) * field_amplitudes
        ionization_rates[np.isnan(ionization_rates)] = 0
        
        # Ensure the sum is not zero to avoid normalization issues
        if np.sum(ionization_rates) <= 0:
            # If all rates are zero, use a uniform distribution
            ionization_rates = np.ones_like(ionization_rates) / len(ionization_rates)
        else:
            # Normalize
            ionization_rates = ionization_rates / np.sum(ionization_rates)
        
        # Select ionization time based on probability distribution
        ionization_time_index = np.random.choice(len(times), p=ionization_rates)
        return times[ionization_time_index]
    
    def _initial_conditions(self, t_ion):
        """
        Generate initial conditions for the electron at the ionization time.
        """
        # Get the field at ionization time
        Ex, Ey, Ez = laser_field(
            t_ion, self.E0, self.omega, self.pulse_duration_au,
            self.cep, self.ellipticity, self.polarization_angle, self.pulse_shape
        )
        
        # Total field amplitude and direction
        field_amp = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        
        if field_amp < 1e-10:
            # If field is too weak, use default initial position
            x0, y0, z0 = 0.1, 0.1, 0.1
        else:
            # Normalized field direction
            direction = np.array([Ex, Ey, Ez]) / field_amp
            
            # Tunnel exit along field direction
            tunnel_distance = 1.0 / field_amp
            
            # Initial position along the tunnel exit
            x0 = -direction[0] * tunnel_distance
            y0 = -direction[1] * tunnel_distance
            z0 = -direction[2] * tunnel_distance
        
        # Initial momentum (tunneling theory - perpendicular components from quantum uncertainty)
        # First, find perpendicular directions to the field
        if field_amp < 1e-10:
            # Default directions if field is too weak
            perp1 = np.array([1, 0, 0])
            perp2 = np.array([0, 1, 0])
        else:
            direction = np.array([Ex, Ey, Ez]) / field_amp
            
            # Find two perpendicular directions
            if abs(direction[2]) < 0.9:
                perp1 = np.array([direction[1], -direction[0], 0])
            else:
                perp1 = np.array([1, 0, -direction[0]/direction[2]])
                
            perp1 = perp1 / np.sqrt(np.sum(perp1**2))  # Normalize
            perp2 = np.cross(direction, perp1)  # Second perpendicular direction
        
        # Sample perpendicular momentum components from a Gaussian distribution
        p_perp1 = np.random.normal(0, 0.1)
        p_perp2 = np.random.normal(0, 0.1)
        
        # Initial momentum (perpendicular components only, along-field component is 0)
        px0 = p_perp1 * perp1[0] + p_perp2 * perp2[0]
        py0 = p_perp1 * perp1[1] + p_perp2 * perp2[1]
        pz0 = p_perp1 * perp1[2] + p_perp2 * perp2[2]
        
        return np.array([x0, y0, z0, px0, py0, pz0])
    
    def _compute_single_trajectory(self, seed):
        """
        Compute a single electron trajectory.
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (trajectory_solution, ionization_time, final_momentum, quantum_phase)
        """
        try:
            # Set seed for reproducibility
            np.random.seed(seed)
            
            # Generate ionization time
            t_ion = self._generate_ionization_time()
            
            # Set initial conditions
            y0 = self._initial_conditions(t_ion)
            
            # Integrate equations of motion from ionization time
            t_span = [t_ion, self.t_max]
            sol = solve_ivp(
                self._equations_of_motion, 
                t_span, 
                y0, 
                method='RK45', 
                t_eval=np.linspace(t_ion, self.t_max, 100),
                atol=1e-10,
                rtol=1e-8
            )
            
            # Get final momentum - with robust handling of different sol.y formats
            if sol.success:
                try:
                    # Try to get the final state safely
                    if hasattr(sol, 'y') and sol.y is not None:
                        y_array = np.asarray(sol.y)
                        
                        # Check the dimensionality of the array
                        if y_array.ndim == 2:
                            # Normal case: y_array is 2D with shape (state_vars, time_points)
                            final_state = y_array[:, -1]
                            if len(final_state) >= 6:
                                final_momentum = final_state[3:6]  # px, py, pz
                                
                                # Calculate quantum phase
                                quantum_phase = self._calculate_quantum_phase(sol, t_ion, y0)
                                
                                return sol, t_ion, final_momentum, quantum_phase
                        elif y_array.ndim == 1:
                            # Special case: y_array is 1D, meaning only one time point
                            # or one state variable (shouldn't happen but let's handle it)
                            if len(y_array) >= 6:
                                # Assume it's a single time point with all state variables
                                final_momentum = y_array[3:6]
                                
                                # Can't calculate quantum phase properly with only one point
                                quantum_phase = 0.0
                                
                                return sol, t_ion, final_momentum, quantum_phase
                    
                    # If we get here, something was wrong with the solution structure
                    print(f"Unusual solution structure for seed {seed}: y.shape={np.asarray(sol.y).shape}")
                    return None, t_ion, np.zeros(3), 0.0
                    
                except Exception as e:
                    print(f"Error extracting final state for seed {seed}: {str(e)}")
                    return None, t_ion, np.zeros(3), 0.0
            else:
                # If integration failed, return None for this trajectory
                print(f"Integration failed for seed {seed}: {sol.message}")
                return None, t_ion, np.zeros(3), 0.0
        except Exception as e:
            # Catch any errors and return None for this trajectory
            print(f"Error in trajectory calculation with seed {seed}: {str(e)}")
            return None, t_ion if 't_ion' in locals() else 0, np.zeros(3), 0.0
    
    def run_simulation(self, parallel=True):
        """
        Run the QTMC simulation.
        
        Parameters:
        -----------
        parallel : bool
            Whether to use parallel computing
        """
        print(f"Running QTMC simulation with {self.num_trajectories} trajectories...")
        print(f"Laser parameters: I={self.laser_intensity:.1e} W/cm², λ={self.laser_wavelength} nm, τ={self.pulse_duration} fs")
        print(f"Pulse shape: {self.pulse_shape}, Ellipticity: {self.ellipticity}, Polarization angle: {self.polarization_angle:.2f} rad")
        print(f"Atom parameters: Ip={self.Ip} a.u., Z={self.Z}")
        
        # Clear previous results
        self.trajectories = []
        self.ionization_times = []
        self.final_momenta = []
        self.quantum_phases = []
        
        if parallel and self.num_trajectories > 1:
            print(f"Using {self.n_processes} CPU cores for parallel computation")
            
            # Create a pool of worker processes
            with mp.Pool(processes=self.n_processes) as pool:
                # Generate random seeds for reproducibility
                seeds = np.random.randint(0, 2**32, size=self.num_trajectories)
                
                # Run trajectories in parallel
                results = []
                for result in tqdm(
                    pool.imap_unordered(self._compute_single_trajectory, seeds),
                    total=self.num_trajectories,
                    desc="Computing trajectories"
                ):
                    results.append(result)
                
                # Unpack results, skipping any that failed
                for sol, t_ion, final_momentum, quantum_phase in results:
                    if sol is not None:
                        self.trajectories.append(sol)
                        self.ionization_times.append(t_ion)
                        self.final_momenta.append(final_momentum)
                        self.quantum_phases.append(quantum_phase)
        else:
            # Sequential computation
            for i in tqdm(range(self.num_trajectories), desc="Computing trajectories"):
                # Set seed for reproducibility
                seed = np.random.randint(0, 2**32)
                sol, t_ion, final_momentum, quantum_phase = self._compute_single_trajectory(seed)
                
                # Store results if successful
                if sol is not None:
                    self.trajectories.append(sol)
                    self.ionization_times.append(t_ion)
                    self.final_momenta.append(final_momentum)
                    self.quantum_phases.append(quantum_phase)
        
        # Calculate ionization probability (simplified)
        self.ionization_prob = len(self.trajectories) / self.num_trajectories
        
        print(f"Simulation completed successfully with {len(self.trajectories)} valid trajectories.")
    
    def plot_laser_field(self, save_path=None):
        """
        Plot the laser electric field used in the simulation.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        plot_laser_field(
            self.E0, self.omega, self.pulse_duration_au,
            self.cep, self.ellipticity, self.polarization_angle, self.pulse_shape,
            t_max=self.t_max, save_path=save_path
        )
        
    def momentum_distribution(self):
        """
        Calculate the momentum distribution of ionized electrons.
        """
        if not self.final_momenta:
            raise ValueError("No simulation results available. Run simulation first.")
            
        momenta = np.array(self.final_momenta)
        return momenta
    
    def calculate_momentum_spectrum(self, px_range=(-1.5, 1.5), pz_range=(-1.5, 1.5), grid_size=200):
        """
        Calculate the quantum-mechanically correct momentum spectrum using grid-based binning
        with coherent summation of quantum phases.
        
        Parameters:
        -----------
        px_range : tuple
            Range of px values to include (min, max)
        pz_range : tuple
            Range of pz values to include (min, max)
        grid_size : int
            Number of grid points in each dimension
            
        Returns:
        --------
        tuple
            (px_grid, pz_grid, spectrum)
            Where px_grid and pz_grid are meshgrid arrays defining the grid points,
            and spectrum is the 2D momentum spectrum
        """
        if not self.final_momenta or not self.quantum_phases:
            raise ValueError("No trajectory data available. Run simulation first.")
            
        # Convert data to numpy arrays
        momenta = np.array(self.final_momenta)
        phases = np.array(self.quantum_phases)
        
        # Create a 2D grid
        px_grid = np.linspace(px_range[0], px_range[1], grid_size)
        pz_grid = np.linspace(pz_range[0], pz_range[1], grid_size)
        px_mesh, pz_mesh = np.meshgrid(px_grid, pz_grid)
        
        # Initialize complex amplitude grid
        complex_amplitude = np.zeros((grid_size, grid_size), dtype=complex)
        
        # Calculate bin widths
        dpx = (px_range[1] - px_range[0]) / grid_size
        dpz = (pz_range[1] - pz_range[0]) / grid_size
        
        # For each trajectory, add its complex amplitude to the appropriate grid cell
        for i in range(len(momenta)):
            px, _, pz = momenta[i]  # Use px and pz components (typically the laser polarization plane)
            phase = phases[i]
            
            # Check if momentum is within grid range
            if (px_range[0] <= px <= px_range[1]) and (pz_range[0] <= pz <= pz_range[1]):
                # Find grid indices
                ix = int((px - px_range[0]) / dpx)
                iz = int((pz - pz_range[0]) / dpz)
                
                # Ensure indices are within bounds
                ix = min(max(ix, 0), grid_size - 1)
                iz = min(max(iz, 0), grid_size - 1)
                
                # Add complex amplitude (exp(i*phase)) to the grid cell
                complex_amplitude[iz, ix] += np.exp(1j * phase)
        
        # Calculate probability density as the modulus squared of the complex amplitude
        spectrum = np.abs(complex_amplitude)**2
        
        return px_mesh, pz_mesh, spectrum
    
    def plot_quantum_momentum_spectrum(self, px_range=(-1.5, 1.5), pz_range=(-0.5, 1.5), 
                                      grid_size=300, log_scale=True, save_path=None):
        """
        Plot the quantum-mechanically correct momentum spectrum with interference effects.
        
        Parameters:
        -----------
        px_range : tuple
            Range of px values to include (min, max)
        pz_range : tuple
            Range of pz values to include (min, max)
        grid_size : int
            Number of grid points in each dimension
        log_scale : bool
            Whether to use logarithmic color scale
        save_path : str, optional
            Path to save the figure
        """
        # Calculate the momentum spectrum
        px_mesh, pz_mesh, spectrum = self.calculate_momentum_spectrum(
            px_range=px_range, pz_range=pz_range, grid_size=grid_size
        )
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot the spectrum
        if log_scale:
            # Add small value to avoid log(0)
            spectrum_log = np.log10(spectrum + 1e-10)
            im = plt.pcolormesh(px_mesh, pz_mesh, spectrum_log, cmap='jet', shading='auto')
            plt.colorbar(im, label='log10(Probability)')
        else:
            im = plt.pcolormesh(px_mesh, pz_mesh, spectrum, cmap='jet', shading='auto')
            plt.colorbar(im, label='Probability')
        
        # Add labels and title
        plt.xlabel('px (a.u.)')
        plt.ylabel('pz (a.u.)')
        plt.title('Quantum Momentum Spectrum')
        
        # Add a circle to indicate the expected photoelectron energy for ATI peaks
        up = self.E0**2 / (4 * self.omega**2)  # Ponderomotive energy
        for n in range(1, 4):  # Plot a few ATI rings
            # Energy = n*ω - Ip - Up
            energy = n * self.omega - self.Ip - up
            if energy > 0:
                momentum = np.sqrt(2 * energy)
                circle = plt.Circle((0, 0), momentum, fill=False, color='white', linestyle='--', alpha=0.7)
                plt.gca().add_patch(circle)
        
        # Make the plot square
        plt.axis('square')
        plt.grid(alpha=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()
        
        return spectrum
    
    def plot_momentum_distribution(self, save_path=None, include_phase=False):
        """
        Plot the momentum distribution in the px-pz plane.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        include_phase : bool, optional
            Whether to include phase information in the plot (color coded)
        """
        if not self.final_momenta:
            raise ValueError("No simulation results available. Run simulation first.")
            
        momenta = np.array(self.final_momenta)
        
        plt.figure(figsize=(10, 8))
        
        if include_phase and len(self.quantum_phases) == len(momenta):
            # Color points based on phase (mod 2π)
            phases = np.array(self.quantum_phases) % (2 * np.pi)
            scatter = plt.scatter(momenta[:, 0], momenta[:, 2], s=2, c=phases, 
                                 cmap='hsv', alpha=0.7, vmin=0, vmax=2*np.pi)
            plt.colorbar(scatter, label='Quantum Phase (rad)')
        else:
            plt.scatter(momenta[:, 0], momenta[:, 2], s=1, alpha=0.5)
        
        plt.xlabel('px (a.u.)')
        plt.ylabel('pz (a.u.)')
        plt.title('Momentum Distribution (px-pz plane)')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()
    
    def plot_phase_distribution(self, save_path=None):
        """
        Plot the distribution of quantum phases.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        if not self.quantum_phases:
            raise ValueError("No phase data available. Run simulation first.")
            
        phases = np.array(self.quantum_phases) % (2 * np.pi)  # Mod 2π for visualization
        
        plt.figure(figsize=(10, 6))
        plt.hist(phases, bins=50, alpha=0.7)
        plt.xlabel('Quantum Phase (rad)')
        plt.ylabel('Count')
        plt.title('Distribution of Quantum Phases')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()
        
    def plot_trajectories(self, num_to_plot=10, save_path=None):
        """
        Plot a subset of electron trajectories.
        """
        if not self.trajectories:
            raise ValueError("No simulation results available. Run simulation first.")
            
        plt.figure(figsize=(12, 10))
        
        indices = np.random.choice(len(self.trajectories), min(num_to_plot, len(self.trajectories)), replace=False)
        
        for idx in indices:
            traj = self.trajectories[idx]
            plt.plot(traj.y[0], traj.y[2], linewidth=1, alpha=0.7)
            
        plt.xlabel('x (a.u.)')
        plt.ylabel('z (a.u.)')
        plt.title(f'Sample of {num_to_plot} Electron Trajectories (x-z plane)')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()

if __name__ == "__main__":
    # Example usage
    simulation = QTMCSimulation(
        num_trajectories=1000,  # More trajectories for better statistics
        laser_intensity=1.0e14,
        laser_wavelength=800,
        pulse_duration=10.0,
        ellipticity=0.0,        # Linear polarization for clearer interference
        pulse_shape='sin2',     # sin² pulse shape
        n_processes=4           # Specify number of processes
    )
    
    # Visualize the laser field
    simulation.plot_laser_field()
    
    # Run with parallel computing
    simulation.run_simulation(parallel=True)
    
    # Plot classical momentum distribution with phase information
    simulation.plot_momentum_distribution(include_phase=True)
    
    # Plot quantum momentum spectrum with interference effects
    simulation.plot_quantum_momentum_spectrum(px_range=(-1.5, 1.5), pz_range=(-0.5, 1.5), 
                                            grid_size=300, log_scale=True)
    
    # Plot phase distribution
    simulation.plot_phase_distribution()
    
    # Plot trajectories
    simulation.plot_trajectories(num_to_plot=10) 