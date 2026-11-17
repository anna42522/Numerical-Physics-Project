"""Plotting helpers: Wigner function."""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def plot_wigner(state, ax=None, lim=5.0, n=200, title=None):
    xvec = np.linspace(-lim, lim, n)
    W = qt.wigner(state, xvec, xvec)
    ax = ax or plt.gca()
    vmax = np.abs(W).max()
    im = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Re(α)")
    ax.set_ylabel("Im(α)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return im





def plot_results(times, purities, means, parities, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(times, purities, label='Purity', color='blue')
    plt.ylabel('Purity')
    plt.title('Purity over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, means, label='Mean', color='orange')
    plt.ylabel('Mean')
    plt.title('Mean over Time')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(times, parities, label='Parity', color='green')
    plt.xlabel('Time')
    plt.ylabel('Parity')
    plt.title('Parity over Time')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)


def plot_convergence(N_traj_values, mean_errors, std_errors, path):
    """Plot the MCWF convergence error against the number of trajectories."""

    import numpy as np
    import matplotlib.pyplot as plt

    N_traj_values = np.asarray(N_traj_values)
    mean_errors = np.asarray(mean_errors)
    std_errors = np.asarray(std_errors)

    # Theoretical O(M^-1/2) reference
    theoretical = (
        mean_errors[0]
        * (N_traj_values / N_traj_values[0])**(-0.5)
    )

    # Figure
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Numerical results
    ax.errorbar(
        N_traj_values,
        mean_errors,
        yerr=std_errors,
        fmt='o-',
        color='tab:blue',
        markersize=7,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label='MCWF'
    )

    # Theoretical scaling
    ax.plot(
        N_traj_values,
        theoretical,
        '--',
        color='tab:orange',
        linewidth=2,
        label=r'$O(M^{-1/2})$'
    )

    # Logarithmic axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Number of MCWF trajectories', fontsize=12)
    ax.set_ylabel('Mean Frobenius error', fontsize=12)
    ax.set_title('MCWF convergence', fontsize=14)

    # Grid
    ax.grid(True, which='major', color='gray', alpha=0.4)
    ax.grid(True, which='minor', color='gray', alpha=0.15)

    # Legend
    ax.legend(frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def animate_wigner(
    states,
    times,
    filename="results/figures/wigner_decoherence.gif",
    lim=5.0,
    n_grid=120,
    n_frames=60,
    fps=10,
):
    """
    Creates a smooth animated GIF showing the phase-space Wigner function evolving over time.

    Parameters:
    - states: array of density matrices or state vectors (shape: [N_steps, N_cut, N_cut] or [N_steps, N_cut])
    - times: 1D array of corresponding timestamps
    - filename: path where the .gif file will be saved
    - lim: phase space limit for Re(alpha) and Im(alpha) [-lim, lim]
    - n_grid: grid resolution along each axis (e.g. 120x120)
    - n_frames: maximum number of frames to render (subsamples if len(states) > n_frames)
    - fps: playback speed in frames per second
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import pathlib

    # Ensure parent output directory exists
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Subsample if there are more time steps than desired frames
    total_steps = len(states)
    if total_steps > n_frames:
        frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
    else:
        frame_indices = np.arange(total_steps)

    selected_states = [states[idx] for idx in frame_indices]
    selected_times = [times[idx] for idx in frame_indices]
    pause_frames = [selected_states[0]] * 15
    pause_times = [selected_times[0]] * 15
    selected_states = pause_frames + selected_states
    selected_times = pause_times + selected_times
    # Convert first state to Qobj to set up grid and initial color scaling
    first_state = selected_states[0]
    if not isinstance(first_state, qt.Qobj):
        first_state = qt.Qobj(first_state)

    xvec = np.linspace(-lim, lim, n_grid)
    W0 = qt.wigner(first_state, xvec, xvec)
    # Fix vmax based on initial state so the colorbar remains static across all frames
    vmax = float(np.abs(W0).max())

    fig, ax = plt.subplots(figsize=(6, 5.2))
    plt.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.12)

    # Pre-render Wigner data for all selected frames
    wigner_frames = []
    for s in selected_states:
        qobj = s if isinstance(s, qt.Qobj) else qt.Qobj(s)
        wigner_frames.append(qt.wigner(qobj, xvec, xvec))

    def update(frame_idx):
        ax.clear()
        W = wigner_frames[frame_idx]
        t_val = selected_times[frame_idx]

        im = ax.contourf(
            xvec,
            xvec,
            W,
            80,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel(r"Re($\alpha$)", fontsize=11)
        ax.set_ylabel(r"Im($\alpha$)", fontsize=11)
        ax.set_aspect("equal")
        ax.set_title(rf"Wigner Function $W(\alpha)$ — $t = {t_val:.2f}/\kappa$", fontsize=12)
        return im

    anim = FuncAnimation(fig, update, frames=len(selected_times), interval=1000 / fps)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wigner animation saved successfully to {filename}")
    return filename



