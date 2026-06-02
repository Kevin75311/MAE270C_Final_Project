"""
visualization/animation.py — Time-lapse animation of the ablation procedure.

Produces side-by-side animation of T(x,y,t) and Ω_d(x,y,t).
Output formats: GIF or MP4 (requires ffmpeg for mp4).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh, unflatten, build_region_masks


def animate_ablation(T_history: np.ndarray,
                     Omega_history: np.ndarray,
                     t_vec: np.ndarray,
                     cfg: SimConfig = default_cfg,
                     save_path: str = None,
                     fps: int = None,
                     stride: int = 5) -> animation.FuncAnimation:
    """
    Side-by-side animation: temperature field (left) and damage field (right).

    Parameters
    ----------
    T_history     : (n_steps+1, N)  temperature trajectory
    Omega_history : (n_steps+1, N)  damage trajectory
    t_vec         : (n_steps+1,)    time vector [s]
    stride        : plot every `stride`-th frame (reduces file size)
    save_path     : e.g. 'results/ablation.gif' or 'results/ablation.mp4'
    fps           : frames per second in output video
    """
    if fps is None:
        fps = cfg.viz.animation_fps

    X, Y, _, _, _, _ = build_mesh(cfg)
    cx, cy     = cfg.domain.tumor_center[:2]
    r_tumor    = cfg.domain.tumor_radius

    indices = range(0, len(t_vec), stride)
    n_frames = len(list(indices))

    fig, (ax_T, ax_Om) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(bottom=0.15)

    # ── Initial frame setup ───────────────────────────────────────────────
    norm_T  = Normalize(vmin=cfg.viz.T_display_min, vmax=cfg.viz.T_display_max)
    norm_Om = Normalize(vmin=0.0, vmax=2.0)

    T0  = unflatten(T_history[0], cfg)
    Om0 = unflatten(Omega_history[0], cfg)

    pcm_T  = ax_T.pcolormesh(X*100, Y*100, T0, cmap=cfg.viz.colormap_temperature,
                              norm=norm_T, shading='auto')
    pcm_Om = ax_Om.pcolormesh(X*100, Y*100, Om0, cmap=cfg.viz.colormap_damage,
                               norm=norm_Om, shading='auto')

    fig.colorbar(pcm_T,  ax=ax_T,  label='T [°C]')
    fig.colorbar(pcm_Om, ax=ax_Om, label='Ω_d')

    for ax in (ax_T, ax_Om):
        ax.add_patch(Circle((cx*100, cy*100), r_tumor*100,
                            fill=False, edgecolor='white', linewidth=2, linestyle='--'))
        ax.set_xlabel('x [cm]')
        ax.set_aspect('equal')

    ax_T.set_ylabel('y [cm]')
    ax_T.set_title('Temperature  T(x,y,t)')
    ax_Om.set_title('Damage  Ω_d(x,y,t)')

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')

    # Isoline artists (will be updated each frame)
    iso_T_lines  = [None]
    iso_Om_lines = [None]

    def update(frame_idx):
        k   = list(indices)[frame_idx]
        T_k = unflatten(T_history[k], cfg)
        O_k = unflatten(Omega_history[k], cfg)

        pcm_T.set_array(T_k.ravel())
        pcm_Om.set_array(O_k.ravel())

        # Remove old contours safely
        for coll in ax_T.collections[1:]:
            coll.remove()
        for coll in ax_Om.collections[1:]:
            coll.remove()

        try:
            ax_T.contour(X*100, Y*100, T_k,
                         levels=[cfg.control.T_safe, 60.0],
                         colors=['cyan', 'red'], linewidths=1.5)
        except Exception:
            pass

        try:
            ax_Om.contour(X*100, Y*100, O_k,
                          levels=[1.0], colors=['black'], linewidths=2.5)
        except Exception:
            pass

        time_text.set_text(f't = {t_vec[k]:.1f} s   ({t_vec[k]/60:.1f} min)')
        return [pcm_T, pcm_Om, time_text]

    anim = animation.FuncAnimation(fig, update,
                                   frames=n_frames,
                                   interval=1000.0 / fps,
                                   blit=False)

    if save_path:
        ext = save_path.split('.')[-1].lower()
        if ext == 'gif':
            writer = animation.PillowWriter(fps=fps)
        elif ext == 'mp4':
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer = animation.PillowWriter(fps=fps)
            save_path = save_path + '.gif'

        anim.save(save_path, writer=writer, dpi=cfg.viz.dpi)
        print(f"Animation saved: {save_path}")

    return anim
