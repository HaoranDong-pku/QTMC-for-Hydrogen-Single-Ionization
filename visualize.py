#!/usr/bin/env python3
"""
Visualization tools for QTMC simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

def plot_momentum_spectrum_2d(momenta, plane='pxpz', bins=100, log_scale=True, save_path=None):
    """
    Plot 2D momentum spectrum.
    
    Parameters:
    -----------
    momenta : ndarray
        Array of shape (n_trajectories, 3) containing [px, py, pz] for each trajectory
    plane : str
        Which plane to plot: 'pxpy', 'pxpz', or 'pypz'
    bins : int
        Number of bins for the histogram
    log_scale : bool
        Whether to use logarithmic color scale
    save_path : str, optional
        Path to save the figure
    """
    if plane == 'pxpy':
        x_idx, y_idx = 0, 1
        x_label, y_label = 'p_x (a.u.)', 'p_y (a.u.)'
    elif plane == 'pxpz':
        x_idx, y_idx = 0, 2
        x_label, y_label = 'p_x (a.u.)', 'p_z (a.u.)'
    elif plane == 'pypz':
        x_idx, y_idx = 1, 2
        x_label, y_label = 'p_y (a.u.)', 'p_z (a.u.)'
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    x = momenta[:, x_idx]
    y = momenta[:, y_idx]
    
    plt.figure(figsize=(10, 8))
    
    # Use hexbin for better visualization with many points
    if log_scale:
        plt.hexbin(x, y, gridsize=bins, cmap='inferno', bins='log')
    else:
        plt.hexbin(x, y, gridsize=bins, cmap='inferno')
    
    plt.colorbar(label='Counts')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Momentum Distribution ({plane})')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_quantum_momentum_spectrum(momenta, phases, plane='pxpz', p_range=(-1.5, 1.5), 
                                  grid_size=200, log_scale=True, save_path=None, 
                                  ponderomotive_energy=None, photon_energy=None, 
                                  ionization_potential=None):
    """
    Plot quantum-mechanically correct momentum spectrum with interference effects.
    
    Parameters:
    -----------
    momenta : ndarray
        Array of shape (n_trajectories, 3) containing [px, py, pz] for each trajectory
    phases : ndarray
        Array of quantum phases for each trajectory
    plane : str
        Which plane to plot: 'pxpy', 'pxpz', or 'pypz'
    p_range : tuple or dict
        Range of momentum values to include (min, max) for both axes,
        or dictionary with ranges for each axis: {'px': (-1.5, 1.5), 'py': (-1.5, 1.5), 'pz': (-1.5, 1.5)}
    grid_size : int
        Number of grid points in each dimension
    log_scale : bool
        Whether to use logarithmic color scale
    save_path : str, optional
        Path to save the figure
    ponderomotive_energy : float, optional
        Ponderomotive energy (Up) in a.u. for ATI rings
    photon_energy : float, optional
        Photon energy (ω) in a.u. for ATI rings
    ionization_potential : float, optional
        Ionization potential (Ip) in a.u. for ATI rings
        
    Returns:
    --------
    tuple
        (p1_mesh, p2_mesh, spectrum) where p1 and p2 are the momentum axes in the chosen plane
    """
    # Setup based on chosen plane
    if plane == 'pxpy':
        idx1, idx2 = 0, 1
        label1, label2 = 'p_x (a.u.)', 'p_y (a.u.)'
    elif plane == 'pxpz':
        idx1, idx2 = 0, 2
        label1, label2 = 'p_x (a.u.)', 'p_z (a.u.)'
    elif plane == 'pypz':
        idx1, idx2 = 1, 2
        label1, label2 = 'p_y (a.u.)', 'p_z (a.u.)'
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    # Extract components for the chosen plane
    p1 = momenta[:, idx1]
    p2 = momenta[:, idx2]
    
    # Handle different range formats
    if isinstance(p_range, tuple):
        p1_range = p_range
        p2_range = p_range
    elif isinstance(p_range, dict):
        # Extract ranges for specific components
        if idx1 == 0:
            p1_range = p_range.get('px', (-1.5, 1.5))
        elif idx1 == 1:
            p1_range = p_range.get('py', (-1.5, 1.5))
        else:
            p1_range = p_range.get('pz', (-1.5, 1.5))
            
        if idx2 == 0:
            p2_range = p_range.get('px', (-1.5, 1.5))
        elif idx2 == 1:
            p2_range = p_range.get('py', (-1.5, 1.5))
        else:
            p2_range = p_range.get('pz', (-1.5, 1.5))
    else:
        p1_range = (-1.5, 1.5)
        p2_range = (-1.5, 1.5)
    
    # Create a 2D grid
    p1_grid = np.linspace(p1_range[0], p1_range[1], grid_size)
    p2_grid = np.linspace(p2_range[0], p2_range[1], grid_size)
    p1_mesh, p2_mesh = np.meshgrid(p1_grid, p2_grid)
    
    # Calculate bin widths
    dp1 = (p1_range[1] - p1_range[0]) / grid_size
    dp2 = (p2_range[1] - p2_range[0]) / grid_size
    
    # Initialize complex amplitude grid
    complex_amplitude = np.zeros((grid_size, grid_size), dtype=complex)
    
    # For each trajectory, add its complex amplitude to the appropriate grid cell
    for i in range(len(momenta)):
        mom1, mom2 = p1[i], p2[i]
        phase = phases[i]
        
        # Check if momentum is within grid range
        if (p1_range[0] <= mom1 <= p1_range[1]) and (p2_range[0] <= mom2 <= p2_range[1]):
            # Find grid indices
            i1 = int((mom1 - p1_range[0]) / dp1)
            i2 = int((mom2 - p2_range[0]) / dp2)
            
            # Ensure indices are within bounds
            i1 = min(max(i1, 0), grid_size - 1)
            i2 = min(max(i2, 0), grid_size - 1)
            
            # Add complex amplitude (exp(i*phase)) to the grid cell
            complex_amplitude[i2, i1] += np.exp(1j * phase)
    
    # Calculate probability density as the modulus squared of the complex amplitude
    spectrum = np.abs(complex_amplitude)**2
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot the spectrum
    if log_scale:
        # Add small value to avoid log(0)
        spectrum_log = np.log10(spectrum + 1e-10)
        im = plt.pcolormesh(p1_mesh, p2_mesh, spectrum_log, cmap='jet', shading='auto')
        plt.colorbar(im, label='log10(Probability)')
    else:
        im = plt.pcolormesh(p1_mesh, p2_mesh, spectrum, cmap='jet', shading='auto')
        plt.colorbar(im, label='Probability')
    
    # Add labels and title
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(f'Quantum Momentum Spectrum ({plane})')
    
    # Add circles indicating ATI rings if energy parameters are provided
    if all(x is not None for x in [ponderomotive_energy, photon_energy, ionization_potential]):
        for n in range(1, 4):  # Plot a few ATI rings
            # Energy = n*ω - Ip - Up
            energy = n * photon_energy - ionization_potential - ponderomotive_energy
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
    
    return p1_mesh, p2_mesh, spectrum

def plot_momentum_spectrum_3d(momenta, method='scatter', markers=1000, save_path=None):
    """
    Plot 3D momentum spectrum.
    
    Parameters:
    -----------
    momenta : ndarray
        Array of shape (n_trajectories, 3) containing [px, py, pz] for each trajectory
    method : str
        Visualization method: 'scatter' or 'density'
    markers : int
        Number of points to plot (for scatter) or resolution (for density)
    save_path : str, optional
        Path to save the figure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    px, py, pz = momenta[:, 0], momenta[:, 1], momenta[:, 2]
    
    if method == 'scatter':
        # If we have too many points, sample them
        if len(momenta) > markers:
            idx = np.random.choice(len(momenta), markers, replace=False)
            px, py, pz = px[idx], py[idx], pz[idx]
        
        scatter = ax.scatter(px, py, pz, c=np.sqrt(px**2 + py**2 + pz**2), 
                             cmap='viridis', alpha=0.6, s=2)
        plt.colorbar(scatter, label='|p| (a.u.)')
        
    elif method == 'density':
        # Create a grid in 3D space
        grid_x, grid_y, grid_z = np.mgrid[min(px):max(px):markers*1j,
                                          min(py):max(py):markers*1j,
                                          min(pz):max(pz):markers*1j]
        
        # Stack coordinates
        xyz = np.vstack([px, py, pz])
        
        # Estimate PDF with Gaussian KDE
        kde = gaussian_kde(xyz)
        
        # Evaluate KDE on the grid
        density = kde(np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]))
        
        # Reshape density to grid shape
        density = density.reshape(grid_x.shape)
        
        # Plot isosurfaces at different density levels
        levels = np.linspace(density.min(), density.max(), 5)[1:]
        for level in levels:
            ax.contour(grid_x[:, :, 0], grid_y[:, :, 0], density[:, :, 0], 
                       levels=[level], cmap='viridis', alpha=0.5)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    ax.set_xlabel('p_x (a.u.)')
    ax.set_ylabel('p_y (a.u.)')
    ax.set_zlabel('p_z (a.u.)')
    ax.set_title('3D Momentum Distribution')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_energy_spectrum(momenta, bins=100, save_path=None):
    """
    Plot energy spectrum of ionized electrons.
    
    Parameters:
    -----------
    momenta : ndarray
        Array of shape (n_trajectories, 3) containing [px, py, pz] for each trajectory
    bins : int
        Number of bins for the histogram
    save_path : str, optional
        Path to save the figure
    """
    # Calculate energy for each trajectory (E = p²/2 in atomic units)
    energies = 0.5 * np.sum(momenta**2, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(energies, bins=bins, alpha=0.7, color='royalblue')
    plt.xlabel('Energy (a.u.)')
    plt.ylabel('Counts')
    plt.title('Photoelectron Energy Spectrum')
    plt.grid(alpha=0.3)
    
    # Add a log-scale version on the same figure
    ax2 = plt.gca().twinx()
    counts, bins, _ = ax2.hist(energies, bins=bins, alpha=0, color='red')
    ax2.semilogy(0.5 * (bins[1:] + bins[:-1]), counts, color='red', linestyle='--')
    ax2.set_ylabel('Counts (log scale)', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_angular_distribution(momenta, bins=50, save_path=None):
    """
    Plot angular distribution of ionized electrons.
    
    Parameters:
    -----------
    momenta : ndarray
        Array of shape (n_trajectories, 3) containing [px, py, pz] for each trajectory
    bins : int
        Number of bins for the histogram
    save_path : str, optional
        Path to save the figure
    """
    # Calculate spherical coordinates
    p_mag = np.sqrt(np.sum(momenta**2, axis=1))
    theta = np.arccos(momenta[:, 2] / np.maximum(p_mag, 1e-10))  # Polar angle
    phi = np.arctan2(momenta[:, 1], momenta[:, 0])  # Azimuthal angle
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot theta distribution
    ax1 = fig.add_subplot(131)
    ax1.hist(theta, bins=bins, alpha=0.7, color='royalblue')
    ax1.set_xlabel(r'$\theta$ (rad)')
    ax1.set_ylabel('Counts')
    ax1.set_title('Polar Angle Distribution')
    ax1.grid(alpha=0.3)
    
    # Plot phi distribution
    ax2 = fig.add_subplot(132)
    ax2.hist(phi, bins=bins, alpha=0.7, color='green')
    ax2.set_xlabel(r'$\phi$ (rad)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Azimuthal Angle Distribution')
    ax2.grid(alpha=0.3)
    
    # Plot 2D angular distribution
    ax3 = fig.add_subplot(133, projection='polar')
    h = ax3.hist2d(phi, theta, bins=bins, cmap='inferno')
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_xlabel(r'$\phi$')
    ax3.set_ylabel(r'$\theta$', labelpad=20)
    ax3.set_title('2D Angular Distribution')
    plt.colorbar(h[3], ax=ax3, label='Counts')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_ionization_times(ionization_times, field_function, t_max, dt, save_path=None):
    """
    Plot ionization times alongside the laser electric field.
    
    Parameters:
    -----------
    ionization_times : ndarray
        Array of ionization times for each trajectory
    field_function : callable
        Function that returns the electric field at time t
    t_max : float
        Maximum simulation time
    dt : float
        Time step
    save_path : str, optional
        Path to save the figure
    """
    # Create time array
    time = np.arange(0, t_max, dt)
    
    # Calculate electric field
    field = np.array([field_function(t)[0] for t in time])
    
    plt.figure(figsize=(12, 8))
    
    # Plot electric field
    plt.plot(time, field, 'k-', label='Electric Field', alpha=0.7)
    
    # Plot histogram of ionization times
    counts, bins = np.histogram(ionization_times, bins=50, range=(0, t_max))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.fill_between(bin_centers, 0, counts / counts.max() * field.max(), 
                    alpha=0.5, color='royalblue', label='Ionization Times')
    
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Electric Field (a.u.)')
    plt.title('Ionization Times vs. Laser Field')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_trajectory_animation(trajectories, save_path=None):
    """
    Create an animation of electron trajectories.
    
    Parameters:
    -----------
    trajectories : list
        List of trajectory objects from solve_ivp
    save_path : str, optional
        Path to save the animation
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Cannot import matplotlib.animation. Animation not created.")
        return
        
    # Find the common time range for all trajectories
    min_times = np.array([traj.t[0] for traj in trajectories])
    max_times = np.array([traj.t[-1] for traj in trajectories])
    
    t_start = np.max(min_times)
    t_end = np.min(max_times)
    
    # Get positions for each trajectory at specific times
    num_frames = 100
    times = np.linspace(t_start, t_end, num_frames)
    
    # Extract trajectory data at those times
    traj_data = []
    for traj in trajectories[:50]:  # Limit to 50 trajectories for performance
        x_interp = np.interp(times, traj.t, traj.y[0])
        y_interp = np.interp(times, traj.t, traj.y[1])
        z_interp = np.interp(times, traj.t, traj.y[2])
        traj_data.append((x_interp, y_interp, z_interp))
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    all_x = np.concatenate([traj[0] for traj in traj_data])
    all_y = np.concatenate([traj[1] for traj in traj_data])
    all_z = np.concatenate([traj[2] for traj in traj_data])
    
    ax.set_xlim([np.min(all_x), np.max(all_x)])
    ax.set_ylim([np.min(all_y), np.max(all_y)])
    ax.set_zlim([np.min(all_z), np.max(all_z)])
    
    # Initialize empty lines
    lines = [ax.plot([], [], [], '-', alpha=0.5)[0] for _ in range(len(traj_data))]
    
    # Add atom at origin
    ax.plot([0], [0], [0], 'ro', markersize=10)
    
    # Set labels
    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('y (a.u.)')
    ax.set_zlabel('z (a.u.)')
    ax.set_title('Electron Trajectories')
    
    # Animation function
    def animate(i):
        for j, line in enumerate(lines):
            # Plot trajectory up to current frame
            line.set_data(traj_data[j][0][:i], traj_data[j][1][:i])
            line.set_3d_properties(traj_data[j][2][:i])
        return lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                  interval=50, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
    
    plt.close()
    
    return anim 