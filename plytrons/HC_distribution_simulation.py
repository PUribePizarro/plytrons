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
    hot_carriers_plot
)

sim_delta = [0.5, 1, 1.5, 2, 2.5, 3]  # nm 
sim_k = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]  # direction of propagation
sim_e = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]  # direction of electric field
sim_structure = [2,3,4,5,7]  # number of spheres

for structure in sim_structure:
    for delta_ in sim_delta:
        for pol_idx in range(len(sim_e)):
            efield = EField(
                E0=1,                                 # E-field intensity (V/nm)
                k_hat=bcm.v_normalize(sim_k[pol_idx]),     # Planewave direction
                e_hat=bcm.v_normalize(sim_e[pol_idx])      # E-field vector
            )

            c0 = speed_of_light
            Z0, *_ = physical_constants[ "characteristic impedance of vacuum"]
            Z0 = Z0 * eV          # Free space impedance (e/V-s)

            # Define geometrical parameters
            D = 5
                                    # Diameter of spheres (nm)
            lmax = 10                  # Maximum spherical harmonic index

            D1 = 50                       # Diameter of sphere 1 (nm)
            D2 = 5                       # Diameter of sphere 2 (nm)

            # Define Drude model (silver)
            wp = 9.07 * eV / hbar  # rad/s
            gw = 0.060 * eV / hbar  # rad/s
            eps_b = 4.18

            # Define simulation setup
            eps_h = 1                                   # Dielectric constant of host
            w = np.linspace(2.5, 5.0, 10000) * eV / hbar  # frequencies

            def eps_drude(x):
                return eps_b - (wp**2) / ((2*np.pi*c0 / x*1E6) * ((2*np.pi*c0 / x * 1E6) + 1j * gw))

            # efield = EField(
            #     E0=1,                                 # E-field intensity (V/nm)
            #     k_hat=bcm.v_normalize([1, 0, 0]),     # Planewave direction
            #     e_hat=bcm.v_normalize([0, 1, 0])      # E-field vector
            # )

            # Define objects
            # delta = 1  # gap between spheres (nm)
            d_c = D + delta_  # center-to-center distance (nm)

            # BCM_objects = [

                # Single sphere at origin

                # BCMObject(
                #     label='Sphere1',
                #     diameter=D,
                #     lmax=lmax,
                #     eps=eps_drude,
                #     position=np.array([0, 0, 0])

                # Dimer

            if structure == 2:

                BCM_objects = [
                
                    BCMObject(
                        label='Sphere1',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, d_c/2, 0])
                    ),
                    BCMObject(
                        label='Sphere2',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, -d_c/2, 0])
                    ) 
                ]    

                # Trimer

            if structure == 3:
                BCM_objects = [
                
                    BCMObject(
                        label='Sphere1',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, d_c/2, 0])
                    ),
                    BCMObject(
                        label='Sphere2',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, -d_c/2, 0])
                    ),
                    BCMObject(
                        label='Sphere3',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, 0, -(d_c/2)*np.sqrt(3)])
                    )
                ]

                # Tetrahedral

            if structure == 4:

                BCM_objects = [

                    BCMObject(
                        label='Sphere1',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, d_c, d_c])
                    ),
                    BCMObject(
                        label='Sphere2',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, d_c, -d_c])
                    ),
                    BCMObject(
                        label='Sphere3',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, -d_c, d_c])
                    ),
                    BCMObject(
                        label='Sphere4',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0, -d_c, -d_c])
                    )
                ]

                # Pentamer (planar)

            if structure == 5:
                BCM_objects = [

                    BCMObject(
                        label='Sphere1',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/5)))*np.cos(np.pi/2 + 2*np.pi/5*0), (d_c/(2*np.sin(np.pi/5)))*np.sin(np.pi/2 + 2*np.pi/5*0)])
                    ),
                    BCMObject(
                        label='Sphere2',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/5)))*np.cos(np.pi/2 + 2*np.pi/5*1), (d_c/(2*np.sin(np.pi/5)))*np.sin(np.pi/2 + 2*np.pi/5*1)])
                    ),
                    BCMObject(
                        label='Sphere3',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/5)))*np.cos(np.pi/2 + 2*np.pi/5*2), (d_c/(2*np.sin(np.pi/5)))*np.sin(np.pi/2 + 2*np.pi/5*2)])
                    ),
                    BCMObject(
                        label='Sphere4',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/5)))*np.cos(np.pi/2 + 2*np.pi/5*3), (d_c/(2*np.sin(np.pi/5)))*np.sin(np.pi/2 + 2*np.pi/5*3)])
                    ),
                    BCMObject(
                        label='Sphere5',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/5)))*np.cos(np.pi/2 + 2*np.pi/5*4), (d_c/(2*np.sin(np.pi/5)))*np.sin(np.pi/2 + 2*np.pi/5*4)])
                    )

                ]

            if structure == 7:
            #    Heptaner (planar)
                BCM_objects = [

                    BCMObject(
                        label='Sphere1',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*1), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*1)])
                    ),
                    BCMObject(
                        label='Sphere2',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*2), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*2)])
                    ),
                    BCMObject(
                        label='Sphere3',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*3), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*3)])
                    ),
                    BCMObject(
                        label='Sphere4',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*4), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*4)])
                    ),
                    BCMObject(
                        label='Sphere5',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*5), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*5)])
                    ),
                    BCMObject(
                        label='Sphere6',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,(d_c/(2*np.sin(np.pi/6)))*np.cos(np.pi/2 + 2*np.pi/6*6), (d_c/(2*np.sin(np.pi/6)))*np.sin(np.pi/2 + 2*np.pi/6*6)])
                    ),
                    BCMObject(
                        label='Sphere7',
                        diameter=D,
                        lmax=lmax,
                        eps=eps_drude,
                        position=np.array([0,0,0])

                    )
                ]

            pos = np.array([obj.position for obj in BCM_objects], dtype=float)

            # Number of spheres
            Np = len(BCM_objects)

            # Compute interaction matrices and vector
            Gi = [None] * Np
            G0 = [[None for _ in range(Np)] for _ in range(Np)]
            Sv = [None] * Np

            for in_idx in range(Np):
                # Compute internal matrix
                Gi[in_idx] = bcm.Ginternal(BCM_objects[in_idx])
                
                # Compute external interaction matrix
                for jn_idx in range(Np):
                    G0[in_idx][jn_idx] = bcm.Gexternal(BCM_objects[in_idx], BCM_objects[jn_idx])
                
                # Compute external field coefficients
                Sv[in_idx] = bcm.Efield_coupling(BCM_objects[in_idx], efield)

            #-----------------------------------------------------------------------------------------------------
            # Solve system
            #-----------------------------------------------------------------------------------------------------
            Sw = [None] * Np

            dx_max = lmax * (lmax + 1) + (lmax + 1) - 1

            obj_coef = []
            for coef in range(Np):
                obj_coef.append(np.zeros((dx_max, len(w)), dtype=complex))

            for il in range(len(w)):
                c, Si = bcm.solve_BCM(w[il], eps_h, BCM_objects, efield, Gi, G0, Sv)
                for in_idx in range(Np):
                    obj_coef[in_idx][:, il] = c[in_idx]
                    if il == 0:
                        Sw[in_idx] = np.zeros((len(Si[in_idx]), len(w)), dtype=complex)
                    Sw[in_idx][:, il] = Si[in_idx]

            lam_um = 2*np.pi*3E14/w
            for idx_obj in range(Np):
                BCM_objects[idx_obj].set_coefficients(lam_um, obj_coef[idx_obj])

            # Compute scattering and absorption
            Psca, Pabs = bcm.EM_power(w, eps_h, Gi, G0, BCM_objects)

            #-----------------------------------------------------------------------------------------------------
            # Create Results Folder
            #-----------------------------------------------------------------------------------------------------

            outdir = make_results_folder(BCM_objects, efield)  # e.g., results/trimer_D5.0nm_gap1.0nm_Exkz
            # or with extra info:
            # outdir = make_results_folder(BCM_objects, efield, lmax=lmax, eps_h=eps_h, include_timestamp=True)

            print("Saving to:", outdir)
            plt.savefig(outdir / "absorption_spectrum_sphere1.png", dpi=200)

            fig, ax = plot_spheres(
                pos, radii=None,
                cmap_name="turbo",
                light_dir=(1, -1, 2.2),
                title="Nanocluster geometry",
                k_vec=efield.k_hat, E_vec=efield.e_hat, draw_H=True,
                origin_mode="corner", corner=("min","max","min"),
                corner_inset=0.15, offset_along_k=0.15,
                # --- guardado ---
                outdir=outdir,                # <-- misma carpeta
                fname="geometry.png",         # nombre del archivo
                save_dpi=300,
                save_transparent=False,
                show=True
            )


            # Save per-sphere absorption plots
            for i, obj in enumerate(BCM_objects, start=1):
                fig, ax = plt.subplots(figsize=(6, 4))
                Ri = obj.diameter / 2.0
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


            # Quick geometry + run summary
            with open(outdir / "geometry.txt", "w") as f:
                f.write(f"N particles: {len(BCM_objects)}\n")
                for i, obj in enumerate(BCM_objects, start=1):
                    f.write(
                        f"sphere{i}: label={obj.label}, D={obj.diameter:.3f} nm, "
                        f"pos=({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}) nm, "
                        f"lmax={obj.lmax}\n"
                    )
                f.write(f"eps_h={eps_h}\n")
                f.write(f"E0={efield.E0} V/nm, e_hat={efield.e_hat}, k_hat={efield.k_hat}\n")

            print("All figures and data saved in:", outdir)


            #-----------------------------------------------------------------------------------------------------

            import plytrons.quantum_well as qw

            a =  D/2
            E_matrix = qw.get_bound_states(a)
            A_matrix = qw.get_normalization(a, E_matrix)
            e_states = qw.e_state_assembly(E_matrix, A_matrix)

            # ----------------------------------- Plotting ---------------------------------------------------------------------
            # -------------------------------------------------------------------------------------------------
            # SETUP NOTE:
            # - 'tau_e' defines the time grid of electron/hole lifetimes (fs) used by hot_e_dist; all dynamics
            #   and static slices index into this array. Choose it wide and dense enough for your system.
            # - 'plot_flag' controls which figures are produced: 'static' (only one τ), 'dynamics' (GIF over τ),
            #   or 'both'.
            # - For the static plot, set:
            #     * 'tau'       → the lifetime (fs) you want to visualize; 'idx' finds the closest in 'tau_e'.
            #     * 'dE_factor' → spectral broadening factor (multiples of the ε grid step in hot_carriers_plot).
            #     * 'delta'     → energy window around the Fermi level E_F to display (eV).
            # - This script assumes you already defined: BCM_objects, lam_um (µm), Pabs, a (nm), D (nm),
            #   EF (eV), e_states (quantum levels), outdir (folder), and efield (polarization tag).
            # -------------------------------------------------------------------------------------------------

            # ----------------------------------- Simulation set up---------------------------------------------------------------------
            tau_e = np.linspace(50,1500,59)                                              # hot carrier lifetime range (fs)
            plot_flag = 'both'                                                       # choose: 'static' | 'dynamics' | 'both'

            # -------------------- edit these to set the static plot --------------------
            tau = 500                          # decay period to plot (fs) for the static snapshot
            dE_factor = 5                      # Lorentzian/Gaussian broadening factor (× grid step)
            delta = 4.0                        # energy window around E_F for plots (eV)
            idx = np.abs(tau_e - tau).argmin() # closest index in tau_e to the requested 'tau'
            # ---------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------------------------------------------

            # hot carrier simulation inputs


            for Np in range(len(BCM_objects)):  # loop over nanoparticles in the cluster

                peaks_pos = find_peaks(Pabs[Np])[0]                                    # indices of absorption peaks for NP Np

                print('')
                print('############################')
                print(f'  Nanoparticle number {Np+1}')
                print('############################')
                print('')
                for resonance_peak in range(len(peaks_pos)):  # loop over each absorption resonance of this NP

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
                    Te, Th, Te_raw, Th_raw, Mfi2, E_vals  = hot_e_dist(                       # main solver: time-resolved HC
                        a, hv, EF, tau_e, e_states, X_lm, Pabs_peak                           # returns Te/Th (smoothed), *_raw, |Mfi|^2, energies
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
                        Np=Np, peak=resonance_peak+1, D=D,
                        efield=efield,         # tag polarization in the saved filename
                        out_dir=outdir,        # directory where to save (created if missing)
                        save_dpi=300
                    )

                    if plot_flag in ['dynamics', 'both']:
                        hot_carrier_dynamics_plot(Te, Th, Te_raw, Th_raw,                      # animated dynamics over τ_e
                                                e_states, Np+1, resonance_peak+1, tau_e, D, hv, EF, dE_factor=5, delta=4.0,
                                                fps=6,
                                                out_path=f"{outdir}/hot_carriers_dynamics_Np{Np+1}_Res{resonance_peak+1}.gif")
                    if plot_flag in ['static', 'both']:
                    
                        hot_carriers_plot(Te[idx], Th[idx], Te_raw[idx], Th_raw[idx],          # static snapshot at τ ≈ 'tau'
                                                    e_states, Np+1, resonance_peak+1, tau, D, hv, EF, dE_factor, delta, efield, 
                                                    out_path = f"{outdir}/hot_carriers_static_Np{Np+1}_Res{resonance_peak+1}_tau{tau}.png")

