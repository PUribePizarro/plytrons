# HC_spatial.py  –  Hot Carrier Spatial Distribution
#
# Advanced layer: radial and 2-D angular density of photogenerated hot
# carriers inside a spherical nanoparticle.
#
# Two levels of approximation
# ───────────────────────────
#   hot_e_dist_spatial  – angle-averaged radial profile  (fast)
#       Uses Te_raw[f] from the primary core and weights each (n,l) state
#       by its radial wavefunction density |A_f|² |j_lf(k_f r)|² r².
#
#   hot_e_density_2d    – polar 2-D density ρ(r,θ)      (slower)
#       Computes m-resolved rates by explicit gaunt-coefficient sums
#       and weights by |A_f|² |j_lf(k_f r)|² |Y_lf,mf(θ)|².
#       Captures polarisation-induced angular anisotropy.
#
# Key formulae
# ────────────
#   n_e(r)   = Σ_f Te_raw[f] × |A_f|² × |j_lf(k_f r)|² × r²    [1/(nm·fs)]
#   ρ_e(r)   = n_e(r) / (4π r²)                                   [1/(nm³·fs)]
#   ρ_e(r,θ) = Σ_{f,mf} Rate_e[f,mf] × |A_f|² × |j_lf|² × |Y_lf,mf(θ)|²
#
# Architecture
# ────────────
#   hot_carriers.py  ← PRIMARY CORE : rates Te_raw[f]
#   HC_spatial.py    ← ADVANCED LAYER: rates × ψ²(r[,θ]) → density map

from typing import Tuple
import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as _NList

# ── Primary core ──────────────────────────────────────────────────────────────
from plytrons.hot_carriers import hot_e_dist, _fermi_dirac

# ── Physics helpers ───────────────────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, gaunt_coeff
from plytrons.quantum_well import js_real, ke

__all__ = [
    "hot_e_dist_spatial",
    "hot_e_density_2d",
    "plot_radial_profile",
    "plot_2d_density",
]


# =============================================================================
# Internal helpers — state map
# =============================================================================

def _build_state_map_lorder(e_state):
    """
    Flatten (n, l) states in l-order, matching the internal ordering of
    hot_e_dist (l=0 first, then l=1, ...).

    Returns
    -------
    E_all  : float64[N]    energies [eV]
    l_all  : int64[N]      angular momentum
    A2_all : float64[N]    |A_{nl}|²  [nm⁻³]
    """
    lmax = len(e_state)
    N = sum(e_state[l].Eb.size for l in range(lmax))

    E_all  = np.empty(N, dtype=np.float64)
    l_all  = np.empty(N, dtype=np.int64)
    A2_all = np.empty(N, dtype=np.float64)

    idx = 0
    for l in range(lmax):
        n_l = e_state[l].Eb.size
        E_all[idx:idx+n_l] = e_state[l].Eb.real
        l_all[idx:idx+n_l] = l
        for n in range(n_l):
            A = e_state[l].A[n]
            A2_all[idx+n] = (A * A.conj()).real
        idx += n_l

    return E_all, l_all, A2_all


def _build_m_state_map(e_state):
    """
    Flatten ALL (n, l, m) states.

    Ordering: for each l: for each m in [-l..l]: for each radial node n.
    States with the same (n, l) but different m share E and A — only m differs.

    Returns
    -------
    E_all : float64[Nm]    energies [eV]
    l_all : int64[Nm]      angular momentum
    m_all : int64[Nm]      magnetic quantum number
    A_all : complex128[Nm] radial normalisation A_{nl}
    """
    lmax = len(e_state)
    N = sum((2*l + 1) * e_state[l].Eb.size for l in range(lmax))

    E_all = np.empty(N, dtype=np.float64)
    l_all = np.empty(N, dtype=np.int64)
    m_all = np.empty(N, dtype=np.int64)
    A_all = np.empty(N, dtype=np.complex128)

    idx = 0
    for l in range(lmax):
        n_l = e_state[l].Eb.size
        for m in range(-l, l + 1):
            E_all[idx:idx+n_l] = e_state[l].Eb.real
            l_all[idx:idx+n_l] = l
            m_all[idx:idx+n_l] = m
            A_all[idx:idx+n_l] = e_state[l].A
            idx += n_l

    return E_all, l_all, m_all, A_all


# =============================================================================
# Numba kernel — angle-averaged radial density
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _radial_density_kernel(
    E_all: np.ndarray,   # (N,) float64, l-ordered
    l_all: np.ndarray,   # (N,) int64
    A2_all: np.ndarray,  # (N,) float64, |A|²
    Te_raw: np.ndarray,  # (N,) float64, normalized hot-electron rate [1/fs]
    Th_raw: np.ndarray,  # (N,) float64, normalized hot-hole rate [1/fs]
    E_F: float,
    r: np.ndarray,       # (Nr,) float64, radial grid [nm]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angle-averaged radial density of photogenerated hot carriers.

    n_e(r) = Σ_{f: E_f ≥ E_F} Te_raw[f] × |A_f|² × |j_lf(k_f r)|² × r²

    Units: [1/(nm·fs)]  (integrates over r to give total rate in 1/fs).
    """
    Nr = r.size
    N  = E_all.size

    n_e_r = np.zeros(Nr, dtype=np.float64)
    n_h_r = np.zeros(Nr, dtype=np.float64)

    for f in range(N):
        l_f  = int(l_all[f])
        k_f  = ke(E_all[f])
        A2_f = A2_all[f]

        if E_all[f] >= E_F:
            rate_f = Te_raw[f]
            for ir in range(Nr):
                jv = js_real(l_f, k_f * r[ir])
                n_e_r[ir] += rate_f * A2_f * jv * jv * r[ir] * r[ir]
        else:
            rate_f = Th_raw[f]
            for ir in range(Nr):
                jv = js_real(l_f, k_f * r[ir])
                n_h_r[ir] += rate_f * A2_f * jv * jv * r[ir] * r[ir]

    return n_e_r, n_h_r


# =============================================================================
# Public API — Part 1: radial profile
# =============================================================================

def hot_e_dist_spatial(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_fs,                # float or 1-D array of lifetimes [fs]
    e_state,               # list[QWLevelSet]
    X_lm: np.ndarray,
    Pabs: float,
    Nr: int = 200,
    tau_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Angle-averaged radial density of hot carriers.

    Calls :func:`~plytrons.hot_carriers.hot_e_dist` for the transition rates,
    then weights each (n, l) state by its radial wavefunction density:

        n_e(r) = Σ_f Te_raw[f] × |A_f|² × |j_lf(k_f r)|² × r²   [1/(nm·fs)]
        ρ_e(r) = n_e(r) / (4π r²)                                  [1/(nm³·fs)]

    Both integrals satisfy ∫_0^a (...) dr = Σ_f Te_raw[f].

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    hv_eV : float
        Photon energy [eV].
    E_F : float
        Fermi energy [eV].
    tau_fs : float or ndarray
        Inelastic lifetime(s) [fs].  If array, `tau_idx` selects which to plot.
    e_state : list[QWLevelSet]
        Quantum-well bound states from :mod:`plytrons.quantum_well`.
    X_lm : ndarray
        BCM multipole expansion coefficients at `hv_eV`.
    Pabs : float
        Absorbed power [eV/fs].
    Nr : int
        Number of radial grid points (default 200).
    tau_idx : int
        Which lifetime index to use when `tau_fs` is an array (default 0).

    Returns
    -------
    r : float64[Nr]
        Radial grid [nm].
    n_e_r : float64[Nr]
        Radial hot-electron density  [1/(nm·fs)].
    n_h_r : float64[Nr]
        Radial hot-hole density      [1/(nm·fs)].
    rho_e_r : float64[Nr]
        3-D angle-averaged hot-electron density  [1/(nm³·fs)].
    rho_h_r : float64[Nr]
        3-D angle-averaged hot-hole density      [1/(nm³·fs)].
    E_all : float64[N]
        State energies [eV] in l-order.
    Te_raw : float64[N]
        Normalised hot-electron rates per (n, l) state [1/fs].
    Th_raw : float64[N]
        Normalised hot-hole rates per (n, l) state [1/fs].
    """
    tau_arr = np.atleast_1d(np.asarray(tau_fs, dtype=np.float64))

    # ── Call primary core ────────────────────────────────────────────────────
    _, _, Te_raw_all, Th_raw_all, _, _, _, _, _ = hot_e_dist(
        a_nm, hv_eV, E_F, tau_arr, e_state, X_lm, Pabs
    )

    # Te_raw_all shape: (len(tau_arr), N) — select chosen lifetime
    Te_raw = Te_raw_all[tau_idx]   # (N,) in l-order
    Th_raw = Th_raw_all[tau_idx]

    # ── State map ────────────────────────────────────────────────────────────
    E_all, l_all, A2_all = _build_state_map_lorder(e_state)

    # ── Radial grid ──────────────────────────────────────────────────────────
    r = np.linspace(0.0, a_nm, Nr)

    # ── Numba kernel ─────────────────────────────────────────────────────────
    n_e_r, n_h_r = _radial_density_kernel(E_all, l_all, A2_all, Te_raw, Th_raw, E_F, r)

    # ── 3-D angle-averaged density  ρ(r) = n(r) / (4π r²) ──────────────────
    # At r=0 both n_e and 4πr² → 0; take continuous limit (L'Hôpital → 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rho_e_r = np.where(r > 0, n_e_r / (4.0 * np.pi * r**2), 0.0)
        rho_h_r = np.where(r > 0, n_h_r / (4.0 * np.pi * r**2), 0.0)

    return r, n_e_r, n_h_r, rho_e_r, rho_h_r, E_all, Te_raw, Th_raw


# =============================================================================
# Internal helpers — m-resolved matrix elements  (for hot_e_density_2d)
# =============================================================================

def _compute_Mfi2_m(E_m, l_m, m_m, A_m, a_nm, X_lm):
    """
    Compute the m-resolved squared matrix element |M_{fi}|² for all state pairs.

    Uses the non-interference approximation over EM multipoles:
        |M_{fi,m}|² = Σ_{le} pref²(le) × G²(lf,le,li,-mf,mel,mi)
                               × |X_{le,mel}|² × I²_rad(le,lf,nf,li,ni)
                               × |A_f|² × |A_i|²

    where mel = mf - mi is uniquely fixed by the angular momentum selection rule.

    Parameters
    ----------
    E_m, l_m, m_m : arrays of shape (Nm,)
        Energy, l, m for all m-states.
    A_m : complex128[Nm]
        Radial normalisations.
    a_nm : float
        Sphere radius [nm].
    X_lm : complex128[:]
        BCM multipole coefficients.

    Returns
    -------
    Mfi2 : float64[Nm, Nm]
        |M_{fi}|² for each (kf, ki) pair.
    """
    N_m   = len(E_m)
    le_max = int(np.floor(np.sqrt(X_lm.size) - 1)) + 1

    Nr = 128
    r  = np.linspace(0.0, a_nm, Nr)
    dr = r[1] - r[0] if Nr > 1 else 1.0

    Mfi2 = np.zeros((N_m, N_m), dtype=np.float64)

    A2_m = (A_m * A_m.conj()).real   # |A|²

    for kf in range(N_m):
        lf = int(l_m[kf])
        mf = int(m_m[kf])
        Ef = float(E_m[kf])
        A2_f = float(A2_m[kf])
        k_f  = float(ke(Ef))

        j_lf = js_real(lf, k_f * r)   # (Nr,)

        for ki in range(N_m):
            li  = int(l_m[ki])
            mi  = int(m_m[ki])
            Ei  = float(E_m[ki])
            A2_i = float(A2_m[ki])
            k_i  = float(ke(Ei))

            mel = mf - mi        # unique mel satisfying -mf + mel + mi = 0

            mfi2 = 0.0

            for le in range(1, le_max + 1):
                # Triangle rule
                if le < abs(lf - li) or le > lf + li:
                    continue
                # Even-parity rule
                if (lf + le + li) & 1:
                    continue
                # mel must lie in [-le, le]
                if mel < -le or mel > le:
                    continue

                # Index into X_lm:  lm_to_idx(le, mel) = (le-1)*(le+1) + (mel+le)
                idx_x = (le - 1) * (le + 1) + (mel + le)
                if idx_x >= X_lm.size:
                    continue

                X_val  = X_lm[idx_x]
                X_abs2 = float(X_val.real**2 + X_val.imag**2)
                if X_abs2 < 1e-30:
                    continue

                G = float(gaunt_coeff(lf, le, li, -mf, mel, mi))
                if G * G < 1e-60:
                    continue

                # Radial integral ∫_0^a j_lf(k_f r) r^{le+2} j_li(k_i r) dr
                j_li = js_real(li, k_i * r)   # (Nr,)
                I = 0.0
                for ir in range(Nr):
                    w = 0.5 * dr if (ir == 0 or ir == Nr - 1) else dr
                    I += w * j_lf[ir] * (r[ir] ** (le + 2)) * j_li[ir]

                # pref² = ((1/eps0) sqrt(le/a³) / (2le+1) / a^{le-1})²
                pref2 = ((1.0 / eps0) ** 2 * le / a_nm ** 3
                         / (2 * le + 1) ** 2 / a_nm ** (2 * (le - 1)))

                mfi2 += pref2 * G * G * X_abs2 * I * I * A2_f * A2_i

            Mfi2[kf, ki] = mfi2

    return Mfi2


def _compute_m_rates(E_m, l_m, Mfi2_m, E_F, tau_val, hv_eV, S):
    """
    Compute m-resolved hot-carrier rates from the |M_{fi}|² matrix.

    Parameters
    ----------
    E_m : float64[Nm]
        Energies of all m-states [eV].
    l_m : int64[Nm]
        Angular momentum of each m-state.
    Mfi2_m : float64[Nm, Nm]
        m-resolved squared matrix elements.
    E_F : float
        Fermi energy [eV].
    tau_val : float
        Inelastic lifetime [fs].
    hv_eV : float
        Photon energy [eV].
    S : float
        Normalisation factor from hot_e_dist (Pabs / P_diss).

    Returns
    -------
    Rate_e : float64[Nm]   normalised hot-electron rate per m-state [1/fs].
    Rate_h : float64[Nm]   normalised hot-hole rate per m-state [1/fs].
    """
    N_m = len(E_m)
    gamma_e = hbar / tau_val

    # Fermi-Dirac at T → 0 (consistent with hot_carriers.py default)
    k_B = 8.617333262e-5
    T0  = 1e-4   # K, effectively zero temperature
    fd  = 1.0 / (np.exp((E_m - E_F) / (k_B * T0)) + 1.0)

    Rate_e_raw = np.zeros(N_m)
    Rate_h_raw = np.zeros(N_m)

    prefactor = 4.0 / tau_val

    EE_f = E_m[:, None]     # (Nm, 1)
    EE_i = E_m[None, :]     # (1, Nm)
    fd_f = fd[:, None]
    fd_i = fd[None, :]

    denom_e = (hv_eV - EE_f + EE_i) ** 2 + gamma_e ** 2   # (Nm, Nm)
    denom_h = (hv_eV - EE_i + EE_f) ** 2 + gamma_e ** 2

    TTe = prefactor * fd_i * (1.0 - fd_f) * (
        Mfi2_m / denom_e + Mfi2_m.T / denom_h
    )
    TTh = prefactor * fd_f * (1.0 - fd_i) * (
        Mfi2_m / denom_h + Mfi2_m.T / denom_e
    )

    Rate_e_raw = TTe.sum(axis=1)   # (Nm,)
    Rate_h_raw = TTh.sum(axis=1)

    return S * Rate_e_raw, S * Rate_h_raw


# =============================================================================
# Public API — Part 2: 2-D polar density
# =============================================================================

def hot_e_density_2d(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_fs,                # float or 1-D array — first element is used
    e_state,               # list[QWLevelSet]
    X_lm: np.ndarray,
    Pabs: float,
    Nr: int = 80,
    Ntheta: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Angle-resolved 2-D polar density of hot carriers  ρ(r, θ).

    Computes the m-resolved transition rates using explicit gaunt-coefficient
    sums, then weights each state by its spatial wavefunction density:

        ρ_e(r, θ) = Σ_{f,mf} Rate_e[f,mf] × |A_f|² × |j_lf(k_f r)|²
                                            × |Y_lf,mf(θ)|²

    Units: [1/(nm³·fs)] such that
        2π ∫∫ ρ_e(r,θ) r² sinθ dr dθ = Σ_{f,mf} Rate_e[f,mf].

    The angular pattern captures polarisation-dependent anisotropy, e.g.
    for a z-polarised dipole mode (mel=0) hot carriers concentrate near
    the poles (θ=0, π) compared to the equator.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    hv_eV : float
        Photon energy [eV].
    E_F : float
        Fermi energy [eV].
    tau_fs : float or ndarray
        Inelastic lifetime [fs].  If array, the first element is used.
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    X_lm : ndarray
        BCM multipole expansion coefficients.
    Pabs : float
        Absorbed power [eV/fs].
    Nr : int
        Radial grid points (default 80).
    Ntheta : int
        Angular grid points from 0 to π (default 120).

    Returns
    -------
    r : float64[Nr]
        Radial grid [nm].
    theta : float64[Ntheta]
        Polar angle grid [rad], from 0 (north pole) to π (south pole).
    rho_e : float64[Nr, Ntheta]
        Hot-electron density [1/(nm³·fs)].
    rho_h : float64[Nr, Ntheta]
        Hot-hole density [1/(nm³·fs)].
    """
    from scipy.special import sph_harm

    tau_arr = np.atleast_1d(np.asarray(tau_fs, dtype=np.float64))
    tau_val = float(tau_arr[0])

    # ── Get normalisation factor S from the primary core ─────────────────────
    _, _, _, _, _, _, S, _, _ = hot_e_dist(
        a_nm, hv_eV, E_F, tau_arr, e_state, X_lm, Pabs
    )

    # ── Build m-state arrays ─────────────────────────────────────────────────
    E_m, l_m, m_m, A_m = _build_m_state_map(e_state)
    N_m = len(E_m)

    # ── m-resolved |M_{fi}|² ─────────────────────────────────────────────────
    Mfi2_m = _compute_Mfi2_m(E_m, l_m, m_m, A_m, a_nm, X_lm)

    # ── m-resolved rates ─────────────────────────────────────────────────────
    Rate_e, Rate_h = _compute_m_rates(E_m, l_m, Mfi2_m, E_F, tau_val, hv_eV, S)

    # ── Grids ────────────────────────────────────────────────────────────────
    r     = np.linspace(0.0, a_nm, Nr)
    theta = np.linspace(0.0, np.pi, Ntheta)

    # ── Precompute |Y_lm(θ)|² for each m-state  (scipy, phi=0) ──────────────
    Ylm2 = np.zeros((N_m, Ntheta))
    for k in range(N_m):
        Y = sph_harm(int(m_m[k]), int(l_m[k]), 0.0, theta)
        Ylm2[k] = Y.real ** 2 + Y.imag ** 2   # |Y|² is phi-independent

    # ── Radial wavefunction density |A_k|² |j_lk(k_k r)|² per m-state ───────
    A2_m = (A_m * A_m.conj()).real
    jl2  = np.zeros((N_m, Nr))
    for k in range(N_m):
        k_k  = float(ke(float(E_m[k])))
        l_k  = int(l_m[k])
        jv   = js_real(l_k, k_k * r)     # (Nr,)
        jl2[k] = A2_m[k] * jv ** 2

    # ── Combine:  ρ(r,θ) = Σ_k Rate[k] × jl2[k,r] × Ylm2[k,θ] ─────────────
    # shape: (Nr, Ntheta)
    rho_e = np.einsum('k,kr,kt->rt', Rate_e, jl2, Ylm2)
    rho_h = np.einsum('k,kr,kt->rt', Rate_h, jl2, Ylm2)

    return r, theta, rho_e, rho_h


# =============================================================================
# Visualisation utilities
# =============================================================================

def plot_radial_profile(
    r: np.ndarray,
    n_e_r: np.ndarray,
    n_h_r: np.ndarray,
    rho_e_r: np.ndarray,
    rho_h_r: np.ndarray,
    a_nm: float,
    E_F: float = None,
    hv_eV: float = None,
    ax=None,
):
    """
    Plot the radial hot-carrier density profile.

    Parameters
    ----------
    r : float64[Nr]
        Radial grid [nm].
    n_e_r, n_h_r : float64[Nr]
        Radial density [1/(nm·fs)] (integrates to total rate).
    rho_e_r, rho_h_r : float64[Nr]
        3-D angle-averaged density [1/(nm³·fs)].
    a_nm : float
        Sphere radius [nm] — used to draw the surface marker.
    E_F, hv_eV : float, optional
        Label values for the title.
    ax : matplotlib Axes, optional
        If None, a new 1×2 figure is created.

    Returns
    -------
    fig, axes : matplotlib Figure and (ax_radial, ax_3d) Axes tuple.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig = ax.figure
        axes = (ax,)

    ax_r, ax_3 = axes[0], axes[1]

    # ── Left: radial shell density n(r) ─────────────────────────────────────
    ax_r.plot(r, n_e_r * 1e3, 'r-', lw=1.8, label='Electrons')
    ax_r.plot(r, n_h_r * 1e3, 'b-', lw=1.8, label='Holes')
    ax_r.axvline(a_nm, ls='--', lw=1, color='k', alpha=0.5, label='Surface')
    ax_r.set_xlabel('r  (nm)')
    ax_r.set_ylabel(r'$n(r)$  $[10^{-3}\,\mathrm{nm}^{-1}\,\mathrm{fs}^{-1}]$')
    ax_r.set_title('Radial shell density')
    ax_r.legend()
    ax_r.grid(True, ls=':')
    ax_r.set_xlim(0, a_nm * 1.05)
    ax_r.set_ylim(bottom=0)

    # ── Right: angle-averaged 3-D density ρ(r) ──────────────────────────────
    ax_3.plot(r[1:], rho_e_r[1:] * 1e3, 'r-', lw=1.8, label='Electrons')
    ax_3.plot(r[1:], rho_h_r[1:] * 1e3, 'b-', lw=1.8, label='Holes')
    ax_3.axvline(a_nm, ls='--', lw=1, color='k', alpha=0.5, label='Surface')
    ax_3.set_xlabel('r  (nm)')
    ax_3.set_ylabel(
        r'$\rho(r)$  $[10^{-3}\,\mathrm{nm}^{-3}\,\mathrm{fs}^{-1}]$')
    ax_3.set_title('Angle-averaged 3-D density')
    ax_3.legend()
    ax_3.grid(True, ls=':')
    ax_3.set_xlim(0, a_nm * 1.05)
    ax_3.set_ylim(bottom=0)

    title_parts = []
    if hv_eV is not None:
        title_parts.append(rf'$h\nu = {hv_eV:.2f}$ eV')
    if E_F is not None:
        title_parts.append(rf'$E_F = {E_F:.2f}$ eV')
    if title_parts:
        fig.suptitle('Hot carrier spatial profile   |   ' + ',  '.join(title_parts))

    fig.tight_layout()
    return fig, axes


def plot_2d_density(
    r: np.ndarray,
    theta: np.ndarray,
    rho_e: np.ndarray,
    rho_h: np.ndarray,
    a_nm: float,
    hv_eV: float = None,
    cmap: str = 'inferno',
):
    """
    Plot the 2-D polar hot-carrier density as a cross-section of the sphere.

    The sphere cross-section is shown in the (x, z) plane (z = r cosθ,
    x = r sinθ), with separate panels for electrons and holes.

    Parameters
    ----------
    r : float64[Nr]
    theta : float64[Ntheta]
    rho_e, rho_h : float64[Nr, Ntheta]
        Density from :func:`hot_e_density_2d`.
    a_nm : float
        Sphere radius [nm].
    hv_eV : float, optional
        Photon energy label.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, (ax_e, ax_h) : Figure and Axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Convert to (x, z) Cartesian
    R, TH = np.meshgrid(r, theta, indexing='ij')   # (Nr, Ntheta)
    X = R * np.sin(TH)
    Z = R * np.cos(TH)

    # Mirror for the full circle (left half)
    X_full = np.concatenate([-X[:, ::-1], X[:, 1:]], axis=1)
    Z_full = np.concatenate([ Z[:, ::-1], Z[:, 1:]], axis=1)
    rho_e_full = np.concatenate([rho_e[:, ::-1], rho_e[:, 1:]], axis=1)
    rho_h_full = np.concatenate([rho_h[:, ::-1], rho_h[:, 1:]], axis=1)

    # Mask outside sphere
    mask = X_full**2 + Z_full**2 > a_nm**2
    rho_e_plot = np.where(mask, np.nan, rho_e_full)
    rho_h_plot = np.where(mask, np.nan, rho_h_full)

    vmin = max(1e-30, min(np.nanmin(rho_e_plot[rho_e_plot > 0]),
                           np.nanmin(rho_h_plot[rho_h_plot > 0])))

    fig, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, rho, label, color in [
        (ax_e, rho_e_plot, 'Electrons', 'Reds'),
        (ax_h, rho_h_plot, 'Holes',     'Blues'),
    ]:
        vmax = np.nanmax(rho)
        im = ax.pcolormesh(X_full, Z_full, rho,
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           cmap=cmap, shading='auto')
        # Sphere outline
        phi_c = np.linspace(0, 2 * np.pi, 300)
        ax.plot(a_nm * np.sin(phi_c), a_nm * np.cos(phi_c),
                'w--', lw=0.8, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_xlabel('x  (nm)')
        ax.set_ylabel('z  (nm)')
        ax.set_title(label)
        fig.colorbar(im, ax=ax,
                     label=r'$\rho$  [nm$^{-3}$ fs$^{-1}$]')

    title = 'Hot carrier 2-D density'
    if hv_eV is not None:
        title += rf'  |  $h\nu = {hv_eV:.2f}$ eV'
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_e, ax_h)
