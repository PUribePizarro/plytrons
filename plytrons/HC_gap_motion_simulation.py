# ================================================================
# BCM → Absorption → Hot-carriers → Ne(Δ, τe) colormaps
# + Δ-GIFs per resonance (at fixed τ)
# + Absorption Δ-GIF per sphere with hv vertical lines
# + Combined Absorption + Hot-carrier GIFs per resonance
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import warnings, csv
from pathlib import Path
from scipy.signal import find_peaks
from scipy.constants import hbar, eV, speed_of_light, physical_constants
from numba.core.errors import NumbaTypeSafetyWarning
import imageio.v2 as imageio

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

# ── Project modules ─────────────────────────────────────────────
import plytrons.bcm_sphere as bcm
from plytrons.bcm_sphere import EField, BCMObject
from plytrons.plot_utils import (
    make_results_folder,
    plot_spheres,
    hot_carrier_gap_dynamics_gif,
    absorption_gap_gif,
    hot_carriers_plot,
    convert_raw_hot,
)
from plytrons.math_utils import eps0
from plytrons.hot_carriers import hot_e_dist, hot_e_cdf_per_photon
import plytrons.quantum_well as qw

# ================================================================
# Global simulation controls
# ================================================================
sim_delta = np.linspace(0.2, 3.0, 10)   # nm
sim_k = [[1,0,0]]
sim_e = [[0,1,0]]
sim_structure = [2]                     # try [2,3,4,5,7] if you want more

# electron lifetimes grid (fs)
tau_e = np.linspace(50.0, 1500.0, 59)
# τ used for the Δ-GIFs
tau_fixed = 500.0  # fs
itau = int(np.abs(tau_e - tau_fixed).argmin())

# energy thresholds (relative to EF) for the Δ-maps
eps_list = [0.5, 1.0, 2.0]  # eV
AGG_MODE = "max"            # or "sum"
tau_report = [100.0, 500.0, 1000.0]

# fixed geometry/optics
D = 5.0             # nm
lmax = 10
eps_h = 1.0
w = np.linspace(2.8, 4.2, 200) * eV / hbar
c0 = speed_of_light
Z0, *_ = physical_constants["characteristic impedance of vacuum"]
Z0 = Z0 * eV

# Drude (Ag-like)
wp = 9.07 * eV / hbar
gw = 0.060 * eV / hbar
eps_b = 4.18
def eps_drude(x):
    return eps_b - (wp**2) / ((2*np.pi*c0 / x*1E6) * ((2*np.pi*c0 / x * 1E6) + 1j * gw))

lam_um = 2*np.pi*3E14/w

# Quantum well & energies
a = D/2.0
E_matrix = qw.get_bound_states(a)
A_matrix = qw.get_normalization(a, E_matrix)
e_states = qw.e_state_assembly(E_matrix, A_matrix)

eps_axis = np.linspace(0.0, 4.0, 100)
EF_global = 5.5  # eV

# ================================================================
# Helpers
# ================================================================
def pol_tag_from_idx(p):
    ehat = bcm.v_normalize(sim_e[p])
    khat = bcm.v_normalize(sim_k[p])
    return f"E{bcm.get_axis(ehat)}(k{bcm.get_axis(khat)})"

def agg_init():
    return np.zeros_like(tau_e, dtype=float)

def agg_update(acc, vec):
    return np.maximum(acc, vec) if AGG_MODE == "max" else (acc + vec)

def sphere_positions(structure, d_c):
    if structure == 2:
        return [
            BCMObject("Sphere1", D, lmax, eps_drude, np.array([0, +d_c/2, 0])),
            BCMObject("Sphere2", D, lmax, eps_drude, np.array([0, -d_c/2, 0])),
        ]
    if structure == 3:
        return [
            BCMObject("Sphere1", D, lmax, eps_drude, np.array([0, +d_c/2, 0])),
            BCMObject("Sphere2", D, lmax, eps_drude, np.array([0, -d_c/2, 0])),
            BCMObject("Sphere3", D, lmax, eps_drude, np.array([0, 0, -(d_c/2)*np.sqrt(3)])),
        ]
    if structure == 4:
        return [
            BCMObject("Sphere1", D, lmax, eps_drude, np.array([0, +d_c, +d_c])),
            BCMObject("Sphere2", D, lmax, eps_drude, np.array([0, +d_c, -d_c])),
            BCMObject("Sphere3", D, lmax, eps_drude, np.array([0, -d_c, +d_c])),
            BCMObject("Sphere4", D, lmax, eps_drude, np.array([0, -d_c, -d_c])),
        ]
    if structure == 5:
        R = d_c / (2*np.sin(np.pi/5))
        return [
            BCMObject(f"Sphere{i+1}", D, lmax, eps_drude,
                      np.array([0, R*np.cos(np.pi/2 + 2*np.pi/5*i), R*np.sin(np.pi/2 + 2*np.pi/5*i)]))
            for i in range(5)
        ]
    if structure == 7:
        R = d_c / (2*np.sin(np.pi/6))
        rings = [
            BCMObject(f"Sphere{i}", D, lmax, eps_drude,
                      np.array([0, R*np.cos(np.pi/2 + 2*np.pi/6*i), R*np.sin(np.pi/2 + 2*np.pi/6*i)]))
            for i in range(1,7)
        ]
        center = [BCMObject("Sphere7", D, lmax, eps_drude, np.array([0,0,0]))]
        return rings + center
    raise ValueError(f"Unsupported structure: {structure}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# simple 1D energy clustering to track resonances across Δ
PEAK_ENERGY_TOL_eV = 0.08
def cluster_peak_energies(energies_eV, tol_eV):
    E = np.array(energies_eV, float)
    if E.size == 0:
        return []
    order = np.argsort(E)
    E_sorted = E[order]
    clusters, start = [], 0
    for i in range(1, len(E_sorted)+1):
        if i == len(E_sorted) or (E_sorted[i] - E_sorted[i-1]) > tol_eV:
            members_sorted_idx = np.arange(start, i)
            members_orig_idx = order[members_sorted_idx]
            center = E_sorted[start:i].mean()
            clusters.append({
                'center_eV': float(center),
                'members_idx': members_orig_idx.tolist(),
            })
            start = i
    return clusters

def combined_absorption_hotcarriers_gif(
    energy_eV: np.ndarray,
    Qabs_frames: list,
    hv_lines_per_delta: list,
    Te_tau_frames: list,
    Th_tau_frames: list,
    Te_raw_frames: list,
    Th_raw_frames: list,
    e_states,
    EF: float,
    D: float,
    hv_frames: list,
    gap_values_nm: np.ndarray,
    tau_fs: float,
    dE_factor: float,
    out_path: Path,
    title_prefix: str = "",
    fps: int = 1,
):
    """
    Build a single GIF where each frame is a 1×2 grid:
      [ Absorption vs hv ]  |  [ Hot-carrier distribution (≈hot_carriers_plot) ]

    - Qabs_frames, hv_lines_per_delta are lists over Δ.
    - Te_tau_frames, Th_tau_frames, Te_raw_frames, Th_raw_frames: per Δ (for fixed τ).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Energies in SAME order as Te/Th (like in hot_carriers_plot)
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    n_levels = len(E_all)
    for arr in Te_tau_frames:
        if len(np.asarray(arr)) != n_levels:
            raise ValueError("Length mismatch between Te_tau_frames and E_all.")
    for arr in Th_tau_frames:
        if len(np.asarray(arr)) != n_levels:
            raise ValueError("Length mismatch between Th_tau_frames and E_all.")

    to_ps = 1000.0  # fs → ps
    n_frames = len(gap_values_nm)
    dt = 1.0 / max(fps, 1)

    with imageio.get_writer(out_path, mode="I", duration=dt) as writer:
        for k in range(n_frames):
            gap_nm = float(gap_values_nm[k])
            hv = float(hv_frames[k]) if k < len(hv_frames) else 0.0
            if hv <= 0.0:
                # Fallback: use some representative energy (e.g. average of nonzero hv)
                hv_pos = [h for h in hv_frames if h > 0]
                hv = float(np.mean(hv_pos)) if hv_pos else 1e-6

            # ========= HOT CARRIERS (right panel; same logic as hot_carriers_plot) =========
            Te = np.asarray(Te_tau_frames[k], float) / 1000.0  # 10^-3 nm^-3 → nm^-3
            Th = np.asarray(Th_tau_frames[k], float) / 1000.0
            Te_raw = np.asarray(Te_raw_frames[k], float)
            Th_raw = np.asarray(Th_raw_frames[k], float)

            # Smooth distribution to fine energy grid
            x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)

            # fs^-1 → ps^-1, then per eV
            scale = to_ps / hv
            Te_x *= scale
            Th_x *= scale

            # masks around EF with window Δ = gap_nm
            mask_e = (x >= EF) & (x <= EF + gap_nm)
            mask_h = (x <= EF) & (x >= EF - gap_nm)

            # ========= ABSORPTION (left panel) =========
            Qabs = np.asarray(Qabs_frames[k], float)
            hv_lines = hv_lines_per_delta[k]

            fig, (ax_abs, ax_hc) = plt.subplots(1, 2, figsize=(13, 4.5))

            # ---- Left: Absorption vs hv ----
            ax_abs.plot(energy_eV, Qabs, 'k')
            for hv_line in hv_lines:
                ax_abs.axvline(hv_line, ls=':', lw=0.8, color='gray', alpha=0.6)
            ax_abs.axvline(hv, ls='--', lw=1.2, color='red', alpha=0.95, label=r'resonance $h\nu$')

            ax_abs.set_xlabel('Photon energy (eV)')
            ax_abs.set_ylabel('Absorption efficiency')
            ax_abs.set_title(rf'Absorption · $\Delta={gap_nm:.2f}$ nm')
            ax_abs.grid(True, ls=':')
            ax_abs.legend(loc='upper right')

            # ---- Right: Hot carriers (≈ hot_carriers_plot) ----
            ax2 = ax_hc.twinx()
            ax_hc.fill_between((x - EF)[mask_e], Te_x[mask_e], color='r', alpha=0.38, label='Electrons (dens.)')
            ax_hc.fill_between((x - EF)[mask_h], Th_x[mask_h], color='b', alpha=0.38, label='Holes (dens.)')

            bar_width = 2.0e-2
            ax2.bar(E_all - EF, Te_raw * to_ps, width=bar_width, color='firebrick', alpha=0.9, label='Electrons')
            ax2.bar(E_all - EF, Th_raw * to_ps, width=bar_width, color='royalblue', alpha=0.9, label='Holes')

            ax_hc.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5)
            ax_hc.axvline(+hv, ls='--', lw=1, color='gray', alpha=0.6)
            ax_hc.axvline(-hv, ls='--', lw=1, color='gray', alpha=0.6)

            ax_hc.set_xlim(-gap_nm, +gap_nm)
            ax_hc.set_xlabel('Hot carrier energy relative to Fermi level (eV)')
            ax_hc.set_ylabel(r'Rate density $[10^{-3}\,\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
            ax2.set_ylabel(r'Rate per particle $[\mathrm{ps}^{-1}]$')

            ax_hc.set_ylim(
                0.0,
                1.05 * max(
                    Te_x[mask_e].max(initial=0.0),
                    Th_x[mask_h].max(initial=0.0),
                ),
            )
            ax2.set_ylim(
                0.0,
                3.0 * to_ps * max(
                    Te_raw.max(initial=0.0),
                    Th_raw.max(initial=0.0),
                ),
            )

            h1, l1 = ax_hc.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax_hc.legend(h1 + h2, l1 + l2, loc='upper right')

            fig.suptitle(
                rf'{title_prefix}'
                rf'\n$\Delta={gap_nm:.2f}\,\mathrm{{nm}},\ h\nu={hv:.2f}\,\mathrm{{eV}},\ \tau={tau_fs/1000:.2f}\,\mathrm{{ps}}$'
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])

            # Convert figure to RGB array and append to GIF
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(frame)
            plt.close(fig)


def static_multi_resonance_grid_gif(
    energy_eV: np.ndarray,
    Qabs_frames: list,
    hv_lines_per_delta: list,
    resonances_data: list,
    e_states,
    EF: float,
    gap_values_nm: np.ndarray,
    tau_fs: float,
    dE_factor: float,
    out_path: Path,
    title_prefix: str = "",
    fps: int = 1,
):
    """
    GIF with one absorption panel + up to 3 hot-carrier distributions
    (one per static resonance) for each gap Δ.

    - `resonances_data` is a list of dicts with keys:
        'cid', 'center_eV', 'Te_tau_frames', 'Th_tau_frames',
        'Te_raw_frames', 'Th_raw_frames', 'hv_frames'
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resonances_data:
        return

    # Use at most 3 resonances
    resonances_data = resonances_data[:3]
    hv_static_list = [res["center_eV"] for res in resonances_data]
    n_res = len(resonances_data)

    # Energies in SAME order as Te/Th
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    to_ps = 1000.0  # fs → ps

    n_frames = len(gap_values_nm)
    dt = 1.0 / max(fps, 1)

    with imageio.get_writer(out_path, mode="I", duration=dt) as writer:
        for k in range(n_frames):
            gap_nm = float(gap_values_nm[k])

            # --- figure layout: 1 absorption + n_res distributions ----
            fig, axes = plt.subplots(
                n_res + 1, 1,
                figsize=(8, 3.0 * (n_res + 1)),
                sharex=True
            )
            if n_res == 1:
                axes = [axes]  # make it indexable
            ax_abs = axes[0]
            ax_list = axes[1:]

            # ===== Left/top: Absorption vs hv =====
            Qabs = np.asarray(Qabs_frames[k], float)
            hv_lines_dyn = hv_lines_per_delta[k]

            ax_abs.plot(energy_eV, Qabs, "k")
            # dynamic peak markers (thin gray)
            for hv_line in hv_lines_dyn:
                ax_abs.axvline(hv_line, ls=":", lw=0.8, color="gray", alpha=0.6)

            # static resonance energies (colored)
            colors_static = ["tab:red", "tab:blue", "tab:green"]
            for i, hv_static in enumerate(hv_static_list):
                ax_abs.axvline(
                    hv_static,
                    ls="--",
                    lw=1.4,
                    color=colors_static[i % len(colors_static)],
                    alpha=0.9,
                    label=rf"res {i+1}: {hv_static:.2f} eV",
                )

            ax_abs.set_ylabel("Q$_{abs}$")
            ax_abs.set_title(
                rf"Absorption · $\Delta={gap_nm:.2f}\,\mathrm{{nm}}$"
            )
            ax_abs.grid(True, ls=":")
            ax_abs.legend(loc="upper right")

            # ===== Lower panels: hot-carrier distributions =====
            for i_res, res in enumerate(resonances_data):
                ax = ax_list[i_res]
                hv_dyn = float(res["hv_frames"][k]) if k < len(res["hv_frames"]) else 0.0
                if hv_dyn <= 0.0:
                    hv_dyn = 1e-6  # avoid division by zero

                Te = np.asarray(res["Te_tau_frames"][k], float) / 1000.0
                Th = np.asarray(res["Th_tau_frames"][k], float) / 1000.0

                # Smooth to fine energy grid
                x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)

                # fs^-1 → ps^-1, then per eV
                scale = to_ps / hv_dyn
                Te_x *= scale
                Th_x *= scale

                # window around EF (use same Δ as plot range)
                mask_e = (x >= EF) & (x <= EF + gap_nm)
                mask_h = (x <= EF) & (x >= EF - gap_nm)

                # Filled curves
                ax.fill_between((x - EF)[mask_e], Te_x[mask_e],
                                alpha=0.4, label="e$^-$")
                ax.fill_between((x - EF)[mask_h], Th_x[mask_h],
                                alpha=0.4, label="h$^+$")

                # static resonance lines in (E−EF) coordinates
                hv_static = hv_static_list[i_res]
                ax.axvline(+hv_static, ls="--", lw=1.2,
                           color=colors_static[i_res % len(colors_static)],
                           alpha=0.9)
                ax.axvline(-hv_static, ls="--", lw=1.2,
                           color=colors_static[i_res % len(colors_static)],
                           alpha=0.9)

                ax.set_ylabel(
                    rf"res {i_res+1}  $[10^{{-3}}\,\mathrm{{eV}}^{{-1}}\mathrm{{ps}}^{{-1}}\mathrm{{nm}}^{{-3}}]$"
                )
                ax.set_ylim(bottom=0.0)
                ax.grid(True, ls=":")
                ax.legend(loc="upper right", fontsize=8)

            ax_list[-1].set_xlabel(r"$E - E_F$ (eV)")
            ax_list[-1].set_xlim(-gap_nm, +gap_nm)

            fig.suptitle(
                rf"{title_prefix}"
                rf"\n$\Delta={gap_nm:.2f}\,\mathrm{{nm}},\ \tau={tau_fs/1000:.2f}\,\mathrm{{ps}}$"
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])

            # Convert figure to RGB array and append to GIF
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(frame)
            plt.close(fig)


# ================================================================
# Main sweep
# ================================================================
root_cmaps = Path("results") / "colormaps"
ensure_dir(root_cmaps)

for structure in sim_structure:
    print(f"\n==================== STRUCTURE {structure} ====================")

    # per-structure accumulation (for Ne(Δ, τe) maps)
    Ne_store_by_pol = [{eps: [] for eps in eps_list} for _ in range(len(sim_e))]
    peak_rows_by_pol = [[] for _ in range(len(sim_e))]

    # registry to build Δ-GIFs per resonance (pol → sphere → list of entries)
    registry_by_pol = [[[] for _ in range(20)] for _ in range(len(sim_e))]  # 20 = upper bound; grown if needed

    # registry to build absorption Δ-GIFs per sphere (pol → sphere → list of frames)
    absorb_registry_by_pol = [[[] for _ in range(20)] for _ in range(len(sim_e))]

    struct_out = root_cmaps / f"struct_{structure}"
    ensure_dir(struct_out)

    for d_idx, delta_ in enumerate(sim_delta):
        d_c = D + float(delta_)  # nm

        for pol_idx in range(len(sim_e)):
            efield = EField(
                E0=1.0,
                k_hat=bcm.v_normalize(sim_k[pol_idx]),
                e_hat=bcm.v_normalize(sim_e[pol_idx]),
            )
            pol_tag = pol_tag_from_idx(pol_idx)

            # objects
            BCM_objects = sphere_positions(structure, d_c)
            pos = np.array([o.position for o in BCM_objects], float)
            Np = len(BCM_objects)

            # ensure registries per sphere
            while len(registry_by_pol[pol_idx]) < Np:
                registry_by_pol[pol_idx].append([])
            while len(absorb_registry_by_pol[pol_idx]) < Np:
                absorb_registry_by_pol[pol_idx].append([])

            # BCM matrices
            Gi = [None] * Np
            G0 = [[None] * Np for _ in range(Np)]
            Sv = [None] * Np
            for i in range(Np):
                Gi[i] = bcm.Ginternal(BCM_objects[i])
                for j in range(Np):
                    G0[i][j] = bcm.Gexternal(BCM_objects[i], BCM_objects[j])
                Sv[i] = bcm.Efield_coupling(BCM_objects[i], efield)

            # frequency sweep
            Sw = [None] * Np
            dx_max = lmax * (lmax + 1) + (lmax + 1) - 1
            obj_coef = [np.zeros((dx_max, len(w)), complex) for _ in range(Np)]
            for iw in range(len(w)):
                c, Si = bcm.solve_BCM(w[iw], eps_h, BCM_objects, efield, Gi, G0, Sv)
                for i in range(Np):
                    obj_coef[i][:, iw] = c[i]
                    if iw == 0:
                        Sw[i] = np.zeros((len(Si[i]), len(w)), complex)
                    Sw[i][:, iw] = Si[i]

            for i in range(Np):
                BCM_objects[i].set_coefficients(lam_um, obj_coef[i])
            Psca, Pabs = bcm.EM_power(w, eps_h, Gi, G0, BCM_objects)

            # results folder for this (Δ,pol)
            outdir = make_results_folder(BCM_objects, efield)
            ensure_dir(outdir)

            # geometry image
            _fig, _ax = plot_spheres(
                pos,
                radii=None,
                cmap_name="turbo",
                light_dir=(1, -1, 2.2),
                title=f"Nanocluster geometry · struct={structure} · Δ={delta_:.2f} nm · {pol_tag}",
                k_vec=efield.k_hat,
                E_vec=efield.e_hat,
                draw_H=True,
                origin_mode="corner",
                corner=("min", "max", "min"),
                corner_inset=0.15,
                offset_along_k=0.15,
                outdir=outdir,
                fname="geometry.png",
                save_dpi=300,
                save_transparent=False,
                show=False,
            )
            plt.close(_fig)

            # absorption curves per sphere (+ collect data for Δ-GIF)
            energy_eV = (w * hbar / eV).astype(float)
            for i, obj in enumerate(BCM_objects, start=1):
                fig, ax = plt.subplots(figsize=(6, 4))
                Ri = obj.diameter / 2.0
                Qabs = (Pabs[i - 1] / (efield.E0**2 / (2 * Z0)) / (np.pi * Ri**2)).astype(float)
                ax.plot(energy_eV, Qabs, "k")
                ax.set_title(
                    f"{pol_tag}, sphere{i}, D={obj.diameter:.1f} nm, "
                    f"pos=({obj.position[0]:.1f},{obj.position[1]:.1f},{obj.position[2]:.1f}) nm"
                )
                ax.set_xlabel("Photon energy (eV)")
                ax.set_ylabel("Absorption efficiency")
                ax.grid(True, ls=":")
                fig.tight_layout()
                fig.savefig(outdir / f"absorption_spectrum_sphere{i}.png", dpi=200)
                plt.close(fig)

                # collect for absorption Δ-GIF
                pk_idx = find_peaks(Qabs)[0]
                hv_lines = energy_eV[pk_idx].tolist() if pk_idx.size else []
                absorb_registry_by_pol[pol_idx][i - 1].append(
                    {
                        "delta_idx": int(d_idx),
                        "Qabs": Qabs,
                        "hv_lines": hv_lines,
                    }
                )

            # geometry.txt
            with open(outdir / "geometry.txt", "w") as f:
                f.write(f"structure={structure}\nΔ={delta_:.6f} nm\n")
                f.write(f"pol={pol_tag}\nN particles: {Np}\n")
                for i, obj in enumerate(BCM_objects, start=1):
                    f.write(
                        f"sphere{i}: label={obj.label}, D={obj.diameter:.3f} nm, "
                        f"pos=({obj.position[0]:.3f},{obj.position[1]:.3f},{obj.position[2]:.3f}) nm, "
                        f"lmax={obj.lmax}\n"
                    )
                f.write(
                    f"eps_h={eps_h}\nE0={efield.E0} V/nm, e_hat={efield.e_hat}, k_hat={efield.k_hat}\n"
                )

            # === HOT CARRIERS per resonance (also feed Δ-GIF registry) ===
            Ne_this_delta = {eps: agg_init() for eps in eps_list}
            csv_rows = []
            peak_dir = outdir / "resonances"
            ensure_dir(peak_dir)

            for isph in range(Np):
                peaks_idx = find_peaks(Pabs[isph])[0]
                if peaks_idx.size == 0:
                    continue

                for ipk, ipos in enumerate(peaks_idx, start=1):
                    lam_target = lam_um[ipos]  # µm
                    hv = 2 * np.pi * 3e14 / lam_target * hbar / eV  # eV
                    Pabs_peak = Pabs[isph][ipos] / (np.pi * eps0) * 1e-15  # eV/ps
                    X_lm = BCM_objects[isph].coef_at(lam_target)

                    Te, Th, Te_raw, Th_raw, Mfi2, E_vals = hot_e_dist(
                        a, hv, EF_global, tau_e, e_states, X_lm, Pabs_peak
                    )
                    Ne_sel, _ = hot_e_cdf_per_photon(
                        Te_raw,
                        e_states,
                        hv,
                        Pabs_peak,
                        E_F=EF_global,
                        eps_eval=eps_axis,
                        relative_to_EF=True,
                    )  # (Ntau, N_eps_axis)

                    # save per-peak NPZ + a small plot
                    np.savez(
                        peak_dir / f"sphere{isph+1}_peak{ipk:02d}.npz",
                        tau_e=tau_e,
                        eps_axis=eps_axis,
                        Ne_sel=Ne_sel,
                        lam_target_um=lam_target,
                        hv_eV=hv,
                        Pabs_peak_eVps=Pabs_peak,
                        sphere_index=isph + 1,
                        peak_index=ipk,
                        delta_nm=float(delta_),
                    )
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for eps in eps_list:
                        j = int(np.abs(eps_axis - eps).argmin())
                        ax.plot(tau_e, Ne_sel[:, j], label=rf"$\varepsilon>{eps:.1f}$ eV")
                    ax.set_xlabel(r"$\tau_e$ (fs)")
                    ax.set_ylabel(r"$N_e$ per photon")
                    ax.set_title(
                        f"{pol_tag} · struct={structure} · Δ={delta_:.2f} nm · sphere {isph+1} · peak {ipk}"
                    )
                    ax.legend()
                    ax.grid(True, ls=":")
                    fig.tight_layout()
                    fig.savefig(peak_dir / f"sphere{isph+1}_peak{ipk:02d}_Ne_vs_tau.png", dpi=220)
                    plt.close(fig)

                    # accumulate for Δ-maps
                    row = {
                        "structure": structure,
                        "pol": pol_tag,
                        "delta_nm": float(delta_),
                        "sphere": isph + 1,
                        "peak": ipk,
                        "lambda_nm": float(lam_target * 1e3),
                        "hv_eV": float(hv),
                        "Pabs_peak_eVps": float(Pabs_peak),
                    }
                    for eps in eps_list:
                        j = int(np.abs(eps_axis - eps).argmin())
                        Ne_max = float(Ne_sel[:, j].max())
                        for t_req in tau_report:
                            jt = int(np.abs(tau_e - t_req).argmin())
                            row[f"Ne_eps{eps:.1f}_tau{int(t_req)}fs"] = float(Ne_sel[jt, j])

                        row[f"Ne_eps{eps:.1f}_max_over_tau"] = Ne_max
                        Ne_this_delta[eps] = agg_update(Ne_this_delta[eps], Ne_sel[:, j])
                    csv_rows.append(row)

                    # ====== FEED REGISTRY FOR Δ-GIFs (τ = tau_fixed) ======
                    reg = {
                        "delta_idx": int(d_idx),
                        "hv_eV": float(hv),
                        "Te_tau": Te[itau].astype(float),
                        "Th_tau": Th[itau].astype(float),
                        "Te_raw_tau": Te_raw[itau].astype(float),
                        "Th_raw_tau": Th_raw[itau].astype(float),
                    }
                    registry_by_pol[pol_idx][isph].append(reg)

            # write per-(Δ,pol) CSV of peaks
            if csv_rows:
                with open(outdir / "peaks_summary.csv", "w", newline="") as fcsv:
                    fieldnames = list(csv_rows[0].keys())
                    wcsv = csv.DictWriter(fcsv, fieldnames=fieldnames)
                    wcsv.writeheader()
                    for r in csv_rows:
                        wcsv.writerow(r)
                peak_rows_by_pol[pol_idx].extend(csv_rows)

            # push Ne(τ) per Δ into per-structure store
            for eps in eps_list:
                Ne_store_by_pol[pol_idx][eps].append(Ne_this_delta[eps])

    # ─────────────────────────────────────────────────────────────
    # Per-structure outputs (right after structure finishes)
    # ─────────────────────────────────────────────────────────────
    for pol_idx in range(len(sim_e)):
        pol_tag = pol_tag_from_idx(pol_idx)
        safe_pol = pol_tag.replace("(", "").replace(")", "")
        pol_dir = struct_out / safe_pol
        ensure_dir(pol_dir)

        # global peaks CSV
        rows = peak_rows_by_pol[pol_idx]
        if rows:
            with open(pol_dir / "ALL_peaks_summary.csv", "w", newline="") as fcsv:
                fieldnames = list(rows[0].keys())
                wcsv = csv.DictWriter(fcsv, fieldnames=fieldnames)
                wcsv.writeheader()
                for r in rows:
                    wcsv.writerow(r)

        # Δ-colormaps for each ε threshold
        for eps in eps_list:
            series = Ne_store_by_pol[pol_idx][eps]
            if len(series) != len(sim_delta):
                continue
            Ne_map = np.array(series).T  # (Ntau × NΔ)
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            im = ax.pcolormesh(sim_delta, tau_e, Ne_map, shading="auto", cmap="inferno")
            fig.colorbar(
                im,
                ax=ax,
                label=rf"$N_e(\varepsilon>{eps:.1f}\,\mathrm{{eV}})$ per photon",
            )
            ax.set_xlabel(r"Gap $\Delta$ (nm)")
            ax.set_ylabel(r"Electron lifetime $\tau_e$ (fs)")
            ax.set_title(
                rf"{pol_tag} · structure {structure} · $\varepsilon>{eps:.1f}$ eV · mode={AGG_MODE}"
            )
            fig.tight_layout()
            fig.savefig(pol_dir / f"Ne_map_eps{eps:.1f}eV_{AGG_MODE}.png", dpi=260)
            plt.close(fig)

        # ===== Build Δ-GIFs per resonance (energy clustering) =====
        for isph, entries in enumerate(registry_by_pol[pol_idx], start=1):
            if not entries:
                continue
            sph_dir = pol_dir / f"sphere{isph}"
            ensure_dir(sph_dir)

            # cluster by energy
            E_list = [rec["hv_eV"] for rec in entries]
            clusters = cluster_peak_energies(E_list, PEAK_ENERGY_TOL_eV)
            with open(sph_dir / "resonance_clusters.txt", "w") as fx:
                fx.write(f"# Clustering tol = {PEAK_ENERGY_TOL_eV:.3f} eV\n")
                for cid, cl in enumerate(clusters, start=1):
                    fx.write(
                        f"resonance {cid}: center ~ {cl['center_eV']:.4f} eV, "
                        f"members={len(cl['members_idx'])}\n"
                    )

            # Build absorption frames for this sphere (used by all resonance GIFs)
            energy_eV = (w * hbar / eV).astype(float)
            frames_abs = absorb_registry_by_pol[pol_idx][isph - 1]
            M = energy_eV.size
            zeros_Q = np.zeros(M, float)
            Qabs_frames = []
            hv_lines_frames = []
            for k in range(len(sim_delta)):
                match = None
                for rec in frames_abs:
                    if rec["delta_idx"] == k:
                        match = rec
                        break
                if match is None:
                    Qabs_frames.append(zeros_Q)
                    hv_lines_frames.append([])
                else:
                    Qabs_frames.append(np.asarray(match["Qabs"], float))
                    hv_lines_frames.append(
                        list(map(float, match["hv_lines"]))
                    )

            # energy gridsize for empty frames (use e_states)
            Nlevels = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).size
            zeros = np.zeros(Nlevels, float)

            # will store all resonances for this sphere (for the 3-panel static GIF)
            resonances_data = []

            for cid, cl in enumerate(clusters, start=1):
                # choose the strongest entry per Δ
                by_delta = {}  # d_idx -> entry index
                for idx in cl["members_idx"]:
                    d_idx = entries[idx]["delta_idx"]
                    score = max(
                        entries[idx]["Te_raw_tau"].max(),
                        entries[idx]["Th_raw_tau"].max(),
                    )
                    if (
                        d_idx not in by_delta
                        or score
                        > max(
                            entries[by_delta[d_idx]]["Te_raw_tau"].max(),
                            entries[by_delta[d_idx]]["Th_raw_tau"].max(),
                        )
                    ):
                        by_delta[d_idx] = idx

                # assemble full-length frames (fill holes with zeros)
                Te_tau_frames, Th_tau_frames = [], []
                Te_raw_frames, Th_raw_frames = [], []
                hv_frames = []
                for k in range(len(sim_delta)):
                    if k in by_delta:
                        rec = entries[by_delta[k]]
                        Te_tau_frames.append(rec["Te_tau"])
                        Th_tau_frames.append(rec["Th_tau"])
                        Te_raw_frames.append(rec["Te_raw_tau"])
                        Th_raw_frames.append(rec["Th_raw_tau"])
                        hv_frames.append(rec["hv_eV"])
                    else:
                        Te_tau_frames.append(zeros)
                        Th_tau_frames.append(zeros)
                        Te_raw_frames.append(zeros)
                        Th_raw_frames.append(zeros)
                        hv_frames.append(1e-6)

                # store for the multi-resonance static GIF
                resonances_data.append(
                    {
                        "cid": cid,
                        "center_eV": cl["center_eV"],
                        "Te_tau_frames": Te_tau_frames,
                        "Th_tau_frames": Th_tau_frames,
                        "Te_raw_frames": Te_raw_frames,
                        "Th_raw_frames": Th_raw_frames,
                        "hv_frames": hv_frames,
                    }
                )

                # ===== existing per-resonance GIFs =====
                title_prefix = (
                    f"{pol_tag} · structure {structure} · sphere {isph} · "
                    f"resonance {cid} ({cl['center_eV']:.2f} eV)"
                )
                gif_path = (
                    sph_dir
                    / f"gif_resonance_{cid:02d}_center{cl['center_eV']:.2f}eV_tau{int(tau_fixed)}fs.gif"
                )
                hot_carrier_gap_dynamics_gif(
                    Te_tau_frames,
                    Th_tau_frames,
                    Te_raw_frames,
                    Th_raw_frames,
                    e_states,
                    gap_values_nm=sim_delta,
                    hv_list=hv_frames,
                    EF=EF_global,
                    D=D,
                    tau_fs=tau_fixed,
                    dE_factor=5,
                    Ewin=4.0,
                    efield=None,
                    fps=1,
                    out_path=gif_path,
                    title_prefix=title_prefix,
                )

                # existing combined Absorption + HC GIF (following that single peak)
                combined_gif_path = sph_dir / (
                    f"gif_combined_abs_hot_N{isph}_res{cid:02d}_"
                    f"center{cl['center_eV']:.2f}eV_tau{int(tau_fixed)}fs.gif"
                )
                combined_absorption_hotcarriers_gif(
                    energy_eV=energy_eV,
                    Qabs_frames=Qabs_frames,
                    hv_lines_per_delta=hv_lines_frames,
                    Te_tau_frames=Te_tau_frames,
                    Th_tau_frames=Th_tau_frames,
                    Te_raw_frames=Te_raw_frames,
                    Th_raw_frames=Th_raw_frames,
                    e_states=e_states,
                    EF=EF_global,
                    D=D,
                    hv_frames=hv_frames,
                    gap_values_nm=sim_delta,
                    tau_fs=tau_fixed,
                    dE_factor=5,
                    out_path=combined_gif_path,
                    title_prefix=title_prefix,
                    fps=1,
                )

            # ===== NEW: one GIF with 3 static resonances at once =====
            if resonances_data:
                multi_gif_path = sph_dir / (
                    f"gif_static_resonances_grid_N{isph}_"
                    f"first{min(3, len(resonances_data))}.gif"
                )
                static_multi_resonance_grid_gif(
                    energy_eV=energy_eV,
                    Qabs_frames=Qabs_frames,
                    hv_lines_per_delta=hv_lines_frames,
                    resonances_data=resonances_data,
                    e_states=e_states,
                    EF=EF_global,
                    gap_values_nm=sim_delta,
                    tau_fs=tau_fixed,
                    dE_factor=5,
                    out_path=multi_gif_path,
                    title_prefix=f"{pol_tag} · structure {structure} · sphere {isph}",
                    fps=1,
                )

        # ===== Absorption Δ-GIFs per sphere (with hv vertical lines) =====
        energy_eV = (w * hbar / eV).astype(float)
        for isph, frames in enumerate(absorb_registry_by_pol[pol_idx], start=1):
            if not frames:
                continue
            sph_dir = pol_dir / f"sphere{isph}"
            ensure_dir(sph_dir)

            # reassemble frames in Δ order; fill missing with zeros
            M = energy_eV.size
            zeros = np.zeros(M, float)
            Qabs_frames = []
            hv_lines_frames = []
            for k in range(len(sim_delta)):
                match = None
                for rec in frames:
                    if rec["delta_idx"] == k:
                        match = rec
                        break
                if match is None:
                    Qabs_frames.append(zeros)
                    hv_lines_frames.append([])
                else:
                    Qabs_frames.append(np.asarray(match["Qabs"], float))
                    hv_lines_frames.append(
                        list(map(float, match["hv_lines"]))
                    )

            title_prefix = f"{pol_tag} · structure {structure} · sphere {isph}"
            gif_path = sph_dir / "absorption_gap.gif"
            absorption_gap_gif(
                energy_eV=energy_eV,
                Qabs_frames=Qabs_frames,
                gap_values_nm=sim_delta,
                hv_lines_per_delta=hv_lines_frames,
                title_prefix=title_prefix,
                fps=1,
                out_path=gif_path,
            )

    print(
        f"[OK] Structure {structure} finished. Colormaps, peak summaries, Δ-GIFs, "
        f"combined GIFs, and absorption GIFs saved in: {struct_out.resolve()}"
    )
