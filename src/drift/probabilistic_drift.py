import numpy as np

def backward_drift_particles(
    seed_xy, n_particles=1000, n_steps=24, dt_hours=1.0,
    u_current=-0.16, v_current=0.08, u_wind=4.10, v_wind=-1.80,
    wind_factor=0.035, diffusion=0.015, rng=20181207
):
    """Lagrangian Backward Drift (Hindcasting to find Origin)."""
    generator = np.random.default_rng(rng)
    seeds = np.asarray(seed_xy, dtype=np.float64)
    if len(seeds) == 0:
        seeds = np.array([[0.5, 0.5]])
        
    idx = generator.choice(len(seeds), size=n_particles, replace=True)
    pos = seeds[idx].copy() + generator.normal(0, 0.005, (n_particles, 2))

    # Reverse velocities for backward time steps
    ux = -(u_current + wind_factor * u_wind)
    uy = -(v_current + wind_factor * v_wind)
    
    traj = [pos.copy()]
    for _ in range(n_steps):
        pos = pos + dt_hours * np.array([ux, uy]) + generator.normal(0, diffusion, pos.shape)
        traj.append(pos.copy())
        
    return np.stack(traj, axis=1)

def forward_drift_particles(
    seed_xy, n_particles=1000, n_steps=24, dt_hours=1.0,
    u_current=-0.16, v_current=0.08, u_wind=4.10, v_wind=-1.80,
    wind_factor=0.035, diffusion=0.015, rng=20181207
):
    """Lagrangian Forward Drift (Forecasting Future Slick Impact)."""
    generator = np.random.default_rng(rng + 1)
    seeds = np.asarray(seed_xy, dtype=np.float64)
    if len(seeds) == 0:
        seeds = np.array([[0.5, 0.5]])
        
    idx = generator.choice(len(seeds), size=n_particles, replace=True)
    pos = seeds[idx].copy() + generator.normal(0, 0.005, (n_particles, 2))

    # Forward velocities (positive time steps)
    ux = u_current + wind_factor * u_wind
    uy = v_current + wind_factor * v_wind
    
    traj = [pos.copy()]
    for _ in range(n_steps):
        pos = pos + dt_hours * np.array([ux, uy]) + generator.normal(0, diffusion, pos.shape)
        traj.append(pos.copy())
        
    return np.stack(traj, axis=1)

def origin_density(trajectories, grid_size=64, bandwidth=0.04):
    final = trajectories[:, -1, :]
    final = np.clip(final, 0.0, 1.0)
    ys = np.linspace(0, 1, grid_size)
    xs = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    dens = np.zeros((grid_size, grid_size), dtype=np.float64)
    for p in final:
        dens += np.exp(-(((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * bandwidth**2)))
    dens /= (dens.sum() + 1e-12)
    return dens, final
