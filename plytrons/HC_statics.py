import numpy as np
from scipy.constants import hbar, eV, speed_of_light
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
import plytrons.bcm_sphere as bcm
from plytrons.bcm_sphere import EField, BCMObject
from plytrons.plot_utils import make_results_folder
from plytrons.plot_utils import plot_spheres
from plytrons.hot_carriers import hot_e_dist      # main hot-electron generator (Numba)
import warnings
from numba.core.errors import NumbaTypeSafetyWarning
warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)  # silence harmless Numba warnings
from scipy.signal import find_peaks               # peak finder for absorption spectra
from plytrons.math_utils import eps0              # vacuum permittivity (project's constant)
from plytrons.hot_carriers import hot_e_cdf_per_photon   # CDF of hot-e yields (per photon)
from plytrons.plot_utils import (                 # plotting helpers (project)
    plot_transition_matrix_colormap,
    plot_Ne_cdf_steps,
    hot_carrier_dynamics_plot,
    hot_carriers_plot,
    build_cluster,
    get_polarizations,
)
import plytrons.quantum_well as qw

# =============================================================================
# CONFIGURATION BLOCK — edit here to change the simulation
# =============================================================================
STRUCTURE   = 7          # 2 | 3 | 4 | 5 | 7
D           = 5.0        # sphere diameter (nm)
SIM_GAPS    = [1.0]      # list of gap values (nm) to run
SWEEP_MODE  = None       # None | 'fixed_gap' | 'fixed_centers'
SWEEP_D     = [5, 7, 10, 15, 20]   # diameters for sweep (nm)
FIXED_GAP   = 1.0        # used when SWEEP_MODE == 'fixed_gap'
FIXED_DC    = 8.0        # used when SWEEP_MODE == 'fixed_centers'
lmax        = 10
eps_h       = 1.0
plot_flag   = 'both'     # 'static' | 'dynamics' | 'both'
tau         = 500        # fs — lifetime for static snapshot
dE_factor   = 5          # Lorentzian broadening factor (× grid step)
delta_E     = 4.0        # energy window around E_F for plots (eV)
# =============================================================================

# ── Physical constants ────────────────────────────────────────────────────────
c0          = speed_of_light
Z0, *_      = physical_constants["characteristic impedance of vacuum"]
Z0          = Z0 * eV   # free-space impedance in project units (e/V-s)

# ── Drude model (silver) ──────────────────────────────────────────────────────
wp   = 9.07  * eV / hbar   # plasma  frequency [rad/s]
gw   = 0.060 * eV / hbar   # damping rate      [rad/s]
eps_b = 4.18

def eps_drude(x):
    return eps_b - (wp**2) / ((2*np.pi*c0 / x*1E6) * ((2*np.pi*c0 / x * 1E6) + 1j * gw))

# ── Frequency axis ─────────────────────────────────────────────────────────────
w      = np.linspace(2.5, 5.0, 10000) * eV / hbar   # [rad/s]
lam_um = 2 * np.pi * 3E14 / w                        # [µm]

# ── Hot-carrier lifetime grid ─────────────────────────────────────────────────
tau_e = np.linspace(50, 1500, 59)              # [fs]
idx   = np.abs(tau_e - tau).argmin()           # closest index to requested tau

# ── Build diameter / gap lists from SWEEP_MODE ────────────────────────────────
if SWEEP_MODE is None:
    diameter_list = [D]
    gap_map       = {D: list(SIM_GAPS)}        # {diameter: [gap, ...]}

elif SWEEP_MODE == 'fixed_gap':
    diameter_list = list(SWEEP_D)
    gap_map       = {Dv: [FIXED_GAP] for Dv in diameter_list}

elif SWEEP_MODE == 'fixed_centers':
    diameter_list = list(SWEEP_D)
    gap_map       = {}
    for Dv in diameter_list:
        g = FIXED_DC - Dv
        if g <= 0:
            raise ValueError(
                f"SWEEP_MODE='fixed_centers': gap = FIXED_DC - D = "
                f"{FIXED_DC} - {Dv} = {g} nm <= 0.  "
                "Increase FIXED_DC or reduce the maximum diameter in SWEEP_D."
            )
        gap_map[Dv] = [g]

else:
    raise ValueError(f"SWEEP_MODE={SWEEP_MODE!r} is not recognised. "
                     "Use None, 'fixed_gap', or 'fixed_centers'.")

# ── Polarization list (same for all diameters / gaps) ────────────────────────
polarizations = get_polarizations(STRUCTURE)   # list of (k_hat, e_hat)

# =============================================================================
# MAIN LOOP:  diameters  ×  gaps  ×  polarizations
# =============================================================================
for D_run in diameter_list:
    a        = D_run / 2
    E_matrix = qw.get_bound_states(a)
    A_matrix = qw.get_normalization(a, E_matrix)
    e_states = qw.e_state_assembly(E_matrix, A_matrix)

    for delta_ in gap_map[D_run]:

        # ── Build geometry ────────────────────────────────────────────────
        BCM_objects = build_cluster(STRUCTURE, D_run, delta_, lmax, eps_drude, eps_h)
        Np_cluster  = len(BCM_objects)
        pos         = np.array([obj.position for obj in BCM_objects], dtype=float)

        # ── Precompute geometry matrices (frequency-independent) ──────────
        Gi = [bcm.Ginternal(BCM_objects[i]) for i in range(Np_cluster)]
        G0 = [[bcm.Gexternal(BCM_objects[i], BCM_objects[j])
               for j in range(Np_cluster)] for i in range(Np_cluster)]

        for k_hat, e_hat in polarizations:

            efield = EField(
                E0=1,
                k_hat=bcm.v_normalize(k_hat),
                e_hat=bcm.v_normalize(e_hat),
            )

            Sv = [bcm.Efield_coupling(BCM_objects[i], efield)
                  for i in range(Np_cluster)]

            # ── BCM solve ─────────────────────────────────────────────────
            Sw      = [None] * Np_cluster
            dx_max  = lmax * (lmax + 1) + (lmax + 1) - 1
            obj_coef = [np.zeros((dx_max, len(w)), dtype=complex)
                        for _ in range(Np_cluster)]

            for il in range(len(w)):
                c, Si = bcm.solve_BCM(w[il], eps_h, BCM_objects, efield, Gi, G0, Sv)
                for in_idx in range(Np_cluster):
                    obj_coef[in_idx][:, il] = c[in_idx]
                    if il == 0:
                        Sw[in_idx] = np.zeros((len(Si[in_idx]), len(w)), dtype=complex)
                    Sw[in_idx][:, il] = Si[in_idx]

            for idx_obj in range(Np_cluster):
                BCM_objects[idx_obj].set_coefficients(lam_um, obj_coef[idx_obj])

            # ── Absorption / scattering ───────────────────────────────────
            Psca, Pabs = bcm.EM_power(w, eps_h, Gi, G0, BCM_objects)

            # ── Results folder ────────────────────────────────────────────
            outdir = make_results_folder(BCM_objects, efield)
            print("Saving to:", outdir)
            plt.savefig(outdir / "absorption_spectrum_sphere1.png", dpi=200)

            fig, ax = plot_spheres(
                pos, radii=None,
                cmap_name="turbo",
                light_dir=(1, -1, 2.2),
                title="Nanocluster geometry",
                k_vec=efield.k_hat, E_vec=efield.e_hat, draw_H=True,
                origin_mode="corner", corner=("min", "max", "min"),
                corner_inset=0.15, offset_along_k=0.15,
                outdir=outdir,
                fname="geometry.png",
                save_dpi=300,
                save_transparent=False,
                show=True,
            )

            # Save per-sphere absorption plots
            for i, obj in enumerate(BCM_objects, start=1):
                fig, ax = plt.subplots(figsize=(6, 4))
                Ri   = obj.diameter / 2.0
                Qabs = Pabs[i-1] / (efield.E0**2 / (2 * Z0)) / (np.pi * Ri**2)
                ax.plot(w * hbar / eV, Qabs, 'k')
                ax.set_title(
                    f'E{bcm.get_axis(efield.e_hat)}(k{bcm.get_axis(efield.k_hat)}), '
                    f'sphere{i}, D = {obj.diameter:.1f} nm, '
                    f'(x,y,z) = ({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f}) nm'
                )
                ax.set_xlabel('Photon energy (eV)')
                ax.set_ylabel('Absorption efficiency')
                ax.grid(True, ls=':')
                plt.show()
                fig.tight_layout()
                fig.savefig(outdir / f"absorption_spectrum_sphere{i}.png", dpi=200)
                plt.close(fig)

            # Geometry summary text
            with open(outdir / "geometry.txt", "w") as f:
                f.write(f"N particles: {Np_cluster}\n")
                for i, obj in enumerate(BCM_objects, start=1):
                    f.write(
                        f"sphere{i}: label={obj.label}, D={obj.diameter:.3f} nm, "
                        f"pos=({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}) nm, "
                        f"lmax={obj.lmax}\n"
                    )
                f.write(f"eps_h={eps_h}\n")
                f.write(f"E0={efield.E0} V/nm, e_hat={efield.e_hat}, k_hat={efield.k_hat}\n")

            print("All figures and data saved in:", outdir)

            # ── Hot-carrier loop: per NP × per resonance peak ─────────────
            for Np in range(Np_cluster):

                peaks_pos = find_peaks(Pabs[Np])[0]   # indices of absorption peaks

                print('')
                print('############################')
                print(f'  Nanoparticle number {Np+1}')
                print('############################')
                print('')

                for resonance_peak in range(len(peaks_pos)):

                    lam_target = lam_um[peaks_pos[resonance_peak]]                            # wavelength at max absorption (µm)
                    hv         = 2*np.pi*3E14/lam_target*hbar/eV                              # photon energy (eV): ħω with ω=2πc/λ (λ in µm)
                    EF         = 5.5                                                          # Fermi level (eV) (keep consistent with e_states)
                    Pabs_peak  = Pabs[Np][peaks_pos[resonance_peak]]/(np.pi*eps0)*1e-15       # absorption power at peak (eV/ps)
                    X_lm       = BCM_objects[Np].coef_at(lam_target)                          # EM spherical-harmonic coeffs at λ

                    print(f'Resonance peak number {resonance_peak+1}')
                    print(f'Wavelength at peak absorption: {lam_target*1e3:.2f} nm')          # report λ in nm
                    print(f'Photon energy: {hv:.2f} eV')
                    print(f'Absorption power at peak: {Pabs_peak:.2f} eV/ps')
                    print('-------------------------------------------------------')

                    #------------------------------- Hot Carriers dynamics Simulation -----------------------------------------------

                    from plytrons.hot_carriers import hot_e_dist                              # re-import (redundant but harmless)
                    Te, Th, Te_raw, Th_raw, Mfi2, E_vals, *_ = hot_e_dist(                    # main solver: time-resolved HC
                        a, hv, EF, tau_e, e_states, X_lm, Pabs_peak                           # returns Te/Th (smoothed), *_raw, |Mfi|^2, energies, S, Pabs, P_diss
                    )

                    plot_transition_matrix_colormap(                                          # visualize |M_fi|^2 vs energies
                        Mfi2, E_vals,
                        dedup="sum", scale="asinh", qmin=0.001, qmax=0.999                    # robust contrast with quantile cut
                    )

                    eps_axis = np.array(np.linspace(0.0, 4.0, 100), dtype=float)              # ε thresholds (eV above E_F) for CDF

                    Ne_sel, _ = hot_e_cdf_per_photon(                                         # cumulative yield per photon for ε≥ε_i
                        Te_raw, e_states, hv, Pabs_peak, E_F=EF, eps_eval=eps_axis, relative_to_EF=True
                    )

                    Ne_clean = Ne_sel                                                          # place-holder if you later denoise/smooth

                    plot_Ne_cdf_steps(                                                         # step plot of CDF vs ε with hv markers
                        eps_axis, Ne_clean, tau_e,
                        hv=hv,                 # also draws the 0.2*hv and 0.5*hv vertical lines by default
                        Np=Np, peak=resonance_peak+1, D=D_run,
                        efield=efield,         # tag polarization in the saved filename
                        out_dir=outdir,        # directory where to save (created if missing)
                        save_dpi=300
                    )

                    if plot_flag in ['dynamics', 'both']:
                        hot_carrier_dynamics_plot(Te, Th, Te_raw, Th_raw,                      # animated dynamics over τ_e
                                                e_states, Np+1, resonance_peak+1, tau_e, D_run, hv, EF, dE_factor=5, delta=4.0,
                                                fps=6,
                                                out_path=f"{outdir}/hot_carriers_dynamics_Np{Np+1}_Res{resonance_peak+1}.gif")
                    if plot_flag in ['static', 'both']:

                        hot_carriers_plot(Te[idx], Th[idx], Te_raw[idx], Th_raw[idx],          # static snapshot at τ ≈ 'tau'
                                                    e_states, Np+1, resonance_peak+1, tau, D_run, hv, EF, dE_factor, delta_E, efield,
                                                    out_path = f"{outdir}/hot_carriers_static_Np{Np+1}_Res{resonance_peak+1}_tau{tau}.png")
