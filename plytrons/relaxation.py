# Hot Carrier Relaxation Physics in Plasmonic Nanoparticles
#
# Secondary core: electron-electron and electron-phonon scattering
# following Saavedra, Asenjo-Garcia & Garcia de Abajo,
# ACS Photonics 3, 1637 (2016).
#
# Part A: Electron-electron scattering (eqs 3, 9, 10, 14)
# Part B: Electron-phonon scattering  (γ^{e-ph} = G / c_e(T))
# Part C: Effective lifetime          (1/τ_eff = 1/τ_ee + 1/τ_eph)
#
# Architecture
# ─────────────────────────────────────────────────────────────────
#   hot_carriers.py  ← PRIMARY CORE  : optical excitation (Fermi golden rule)
#   relaxation.py    ← SECONDARY CORE: e-e + e-ph (Saavedra et al. 2016)
#        ↑                   ↑
#   HC_dynamics.py   ← ADVANCED LAYER: simulation + visualization
#   HC_statics.py    ← ADVANCED LAYER: static analysis

from typing import Tuple

import numpy as np
import numba as nb
from numba import prange

# ── Project helpers ────────────────────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, Wigner3j
from plytrons.quantum_well import js_real, ke

# ── Primary core (needed by hot_e_dist_resolved) ──────────────────────────────
from plytrons.hot_carriers import hot_e_dist, _fermi_dirac

__all__ = [
    # Part A – electron-electron scattering
    "ee_scattering_rates", "ee_lifetimes",
    # Part B – electron-phonon scattering
    "electronic_specific_heat", "eph_rate", "eph_lifetime", "eph_rate_vs_temperature",
    "EPH_COUPLING",
    # Part C – combined effective lifetime
    "effective_lifetime", "hot_e_dist_resolved",
]


# #############################################################################
#
#  PART A  –  ELECTRON-ELECTRON SCATTERING
#
#  Reference: Saavedra, Asenjo-Garcia & Garcia de Abajo,
#             ACS Photonics 3, 1637 (2016), equations 3 & 14.
#
#  Key formula (eq 14, m-summed):
#
#    gamma_{j<-i}^{e-e} = (8*pi*k_C * a^5 / hbar) * |A_i|^2 * |A_j|^2
#        * sum_{l_c} angular_factor(l_j, l_c, l_i)
#                    * radial_G_l(l_c, beta_j, l_j, beta_i, l_i, eps)
#                    * [n_T(|omega_ij|) + theta(omega_ij)]
#
#  where:
#    angular_factor = (2*l_j+1)/(4*pi) * W^2(l_j, l_c, l_i, 0, 0, 0)
#    radial_G_l = -Im{ G_l(omega_ij) }   (screened Coulomb Green's function)
#    n_T = Bose-Einstein distribution
#    theta(x) = Heaviside step function
#    omega_ij = E_i - E_j   (energy transfer, in eV)
#
# #############################################################################

# =============================================================================
# A1. Bose-Einstein distribution
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _bose_einstein(dE_eV: float, T_K: float) -> float:
    """
    Bose-Einstein distribution n_T(omega) = 1/(exp(omega/kT) - 1).

    Parameters
    ----------
    dE_eV : float
        Energy transfer |omega| in eV. Must be > 0.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    n_T : float
    """
    k_B = 8.617333262e-5  # eV/K
    if T_K < 1e-6 or dE_eV < 1e-15:
        return 0.0
    arg = dE_eV / (k_B * T_K)
    if arg > 500.0:
        return 0.0
    return 1.0 / (np.exp(arg) - 1.0)


# =============================================================================
# A2. Drude permittivity (Numba-compatible, complex-valued)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _eps_drude(omega_eV: float, wp_eV: float, eps_b: float,
               gamma_eV: float) -> complex:
    """
    Drude dielectric function eps(omega) = eps_b - wp^2 / [omega(omega + i*gamma)].

    Parameters
    ----------
    omega_eV : float
        Photon/transition energy hbar*omega [eV].
    wp_eV : float
        Plasma frequency [eV].
    eps_b : float
        Background (interband) dielectric constant.
    gamma_eV : float
        Damping rate [eV].

    Returns
    -------
    eps : complex128
    """
    omega = omega_eV + 0.0j
    return eps_b - (wp_eV**2) / (omega * (omega + 1j * gamma_eV))


# =============================================================================
# A3. Screened Coulomb Green's function G_l  (SAG16 eq 14)
# =============================================================================
#
# The radial part of the e-e Coulomb interaction inside a dielectric sphere
# involves:
#
#   G_l(omega) = (1/eps) * integral_0^1 f_ij(x) * g_l(x) dx
#              + [(2l+1) / (l*eps + l + 1) - 1/eps] * R_{l+2}
#
# where:
#   f_ij(x) = j_{l_i}(beta_i * x) * j_{l_j}(beta_j * x)
#   g_l(x)  = x^{1-l} * int_0^x y^{l+2} * f(y) dy
#            + x^{l+2} * int_x^1 y^{1-l} * f(y) dy
#   R_{l+2} = int_0^1 x^{l+2} * f(x) dx
#   beta    = k_e * a  (dimensionless wavevector * radius)
#
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _Gl_screened(
    l_c: int,
    beta_j: float,
    l_j: int,
    beta_i: float,
    l_i: int,
    eps_val: complex,
    Nr: int,
) -> complex:
    """
    Screened Coulomb Green's function G_l for a single multipole l_c.

    Parameters
    ----------
    l_c : int
        Coulomb multipole order (>= 1).
    beta_j, l_j : float, int
        Dimensionless wavevector and angular momentum of state j.
    beta_i, l_i : float, int
        Dimensionless wavevector and angular momentum of state i.
    eps_val : complex
        Dielectric function eps(omega_ij).
    Nr : int
        Number of radial grid points.

    Returns
    -------
    G_l : complex128
        The screened Green's function for multipole l_c.
    """
    # radial grid x in [0, 1] (dimensionless r/a)
    x = np.linspace(0.0, 1.0, Nr)
    dx = x[1] - x[0] if Nr > 1 else 1.0

    # f(x) = j_{l_i}(beta_i * x) * j_{l_j}(beta_j * x)
    f = np.empty(Nr, dtype=np.float64)
    for ii in range(Nr):
        f[ii] = js_real(l_i, beta_i * x[ii]) * js_real(l_j, beta_j * x[ii])

    # --- Build g_l(x) via cumulative integrals ---
    # Term 1: I_up(x) = int_0^x y^{l_c+2} f(y) dy   (cumulative from left)
    # Term 2: I_dn(x) = int_x^1 y^{1-l_c} f(y) dy   (cumulative from right)

    # Precompute integrands
    integrand_up = np.empty(Nr, dtype=np.float64)
    integrand_dn = np.empty(Nr, dtype=np.float64)
    for ii in range(Nr):
        xi = x[ii]
        if xi < 1e-30:
            integrand_up[ii] = 0.0
            integrand_dn[ii] = 0.0
        else:
            integrand_up[ii] = (xi ** (l_c + 2)) * f[ii]
            integrand_dn[ii] = (xi ** (1 - l_c)) * f[ii]

    # Cumulative trapezoid from left: I_up[k] = int_0^{x_k} ...
    I_up = np.zeros(Nr, dtype=np.float64)
    for ii in range(1, Nr):
        I_up[ii] = I_up[ii-1] + 0.5 * dx * (integrand_up[ii-1] + integrand_up[ii])

    # Cumulative trapezoid from right: I_dn[k] = int_{x_k}^1 ...
    I_dn = np.zeros(Nr, dtype=np.float64)
    for ii in range(Nr - 2, -1, -1):
        I_dn[ii] = I_dn[ii+1] + 0.5 * dx * (integrand_dn[ii] + integrand_dn[ii+1])

    # g_l(x) = x^{1-l_c} * I_up(x) + x^{l_c+2} * I_dn(x)
    g = np.empty(Nr, dtype=np.float64)
    for ii in range(Nr):
        xi = x[ii]
        if xi < 1e-30:
            g[ii] = 0.0
        else:
            g[ii] = (xi ** (1 - l_c)) * I_up[ii] + (xi ** (l_c + 2)) * I_dn[ii]

    # --- Main integral: int_0^1 f(x) * g_l(x) dx ---
    fg_integral = 0.0
    for ii in range(Nr):
        w_ii = dx
        if ii == 0 or ii == Nr - 1:
            w_ii *= 0.5
        fg_integral += w_ii * f[ii] * g[ii]

    # --- Boundary integral: R_{l+2} = int_0^1 x^{l_c+2} f(x) dx ---
    R_integral = I_up[Nr - 1]  # = int_0^1 x^{l_c+2} f(x) dx

    # --- Assemble G_l ---
    inv_eps = 1.0 / eps_val

    # Boundary correction factor
    boundary_factor = (2*l_c + 1.0) / (l_c * eps_val + l_c + 1.0) - inv_eps

    G_l = inv_eps * fg_integral + boundary_factor * R_integral

    return G_l


# =============================================================================
# A4. Single e-e scattering rate gamma_{j<-i}  (m-summed)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _gamma_ee_pair(
    E_i: float,
    l_i: int,
    At2_i: float,      # |Ã_i|^2  (dimensionless, = |A_i|^2 * a^3)
    E_j: float,
    l_j: int,
    At2_j: float,      # |Ã_j|^2  (dimensionless, = |A_j|^2 * a^3)
    a_nm: float,
    T_K: float,
    wp_eV: float,
    eps_b: float,
    gamma_eV: float,
    Nr: int,
    lc_max: int,
) -> float:
    """
    Compute the e-e scattering rate gamma_{j<-i} (m-summed, in 1/fs).

    Uses dimensionless normalization Ã = A * a^{3/2} to avoid floating-point
    overflow/underflow from mixing large a^5 with small |A|^2 ~ nm^{-3}.

    The rescaled prefactor is:
        8*pi*k_C*a^5/hbar * |A_i|^2 * |A_j|^2
      = 8*pi*k_C/(hbar*a) * |Ã_i|^2 * |Ã_j|^2

    All quantities are O(1) in the dimensionless form.

    Parameters
    ----------
    E_i, l_i, At2_i : float, int, float
        Energy [eV], angular momentum, dimensionless |Ã_i|^2 of initial state.
    E_j, l_j, At2_j : float, int, float
        Energy [eV], angular momentum, dimensionless |Ã_j|^2 of final state.
    a_nm : float
        Sphere radius [nm].
    T_K : float
        Electron temperature [K].
    wp_eV, eps_b, gamma_eV : float
        Drude parameters.
    Nr : int
        Radial grid points.
    lc_max : int
        Maximum Coulomb multipole to sum over.

    Returns
    -------
    gamma : float
        Scattering rate [1/fs].
    """
    # energy transfer
    omega_ij = E_i - E_j   # eV
    abs_omega = abs(omega_ij)

    if abs_omega < 1e-14:
        return 0.0

    # Bose-Einstein + step function factor
    n_T = _bose_einstein(abs_omega, T_K)
    if omega_ij > 0.0:
        thermal_factor = n_T + 1.0   # emission: n_T + 1
    else:
        thermal_factor = n_T          # absorption: n_T

    # Drude permittivity at the transition frequency
    eps_val = _eps_drude(abs_omega, wp_eV, eps_b, gamma_eV)

    # dimensionless wavevectors beta = k_e * a
    beta_i = ke(E_i) * a_nm
    beta_j = ke(E_j) * a_nm

    # Rescaled prefactor: 8*pi*k_C / (hbar * a)
    # k_C = 1/(4*pi*eps0) [eV*nm],  hbar [eV*fs],  a [nm]
    # => [eV*nm] / ([eV*fs] * [nm]) = 1/fs
    # Multiplied by dimensionless |Ã|^2 terms => final units: 1/fs
    k_C = 1.0 / (4.0 * np.pi * eps0)
    prefactor = 8.0 * np.pi * k_C / (hbar * a_nm)

    # sum over Coulomb multipoles l_c
    sum_lc = 0.0
    for l_c in range(1, lc_max + 1):
        # triangle rule: |l_j - l_i| <= l_c <= l_j + l_i
        if l_c < abs(l_j - l_i) or l_c > l_j + l_i:
            continue

        # parity rule: l_j + l_c + l_i must be even
        if ((l_j + l_c + l_i) & 1) == 1:
            continue

        # angular factor (m-summed):
        # sum_{m_j} |gaunt(l_j,l_c,l_i)|^2 / (2l_c+1)
        #   = (2l_j+1)/(4*pi) * W^2(l_j,l_c,l_i,0,0,0)
        W = Wigner3j(l_j, l_c, l_i, 0, 0, 0)
        angular = (2.0 * l_j + 1.0) / (4.0 * np.pi) * (W * W)

        if angular < 1e-30:
            continue

        # screened Coulomb Green's function
        G_l = _Gl_screened(l_c, beta_j, l_j, beta_i, l_i, eps_val, Nr)

        # we need -Im{G_l}
        neg_imag_G = -G_l.imag

        sum_lc += angular * neg_imag_G

    gamma = prefactor * At2_i * At2_j * sum_lc * thermal_factor

    return gamma


# =============================================================================
# A5. Full e-e scattering rate matrix (parallel over states)
# =============================================================================

@nb.njit(fastmath=True, parallel=True)
def _ee_rate_matrix_parallel(
    E_all: np.ndarray,       # float64[:], all bound energies
    l_all: np.ndarray,       # int64[:], angular momentum for each state
    At2_all: np.ndarray,     # float64[:], dimensionless |Ã|^2 = |A|^2 * a^3
    a_nm: float,
    T_K: float,
    wp_eV: float,
    eps_b: float,
    gamma_eV: float,
    Nr: int,
    lc_max: int,
) -> np.ndarray:
    """
    Compute the full e-e scattering rate matrix gamma[j, i] (1/fs).

    gamma[j, i] = rate for electron to scatter from state i into state j.

    Parameters
    ----------
    E_all : float64[N]
        Bound state energies [eV].
    l_all : int64[N]
        Angular momentum quantum number for each state.
    At2_all : float64[N]
        Dimensionless |Ã|^2 = |A|^2 * a^3  for each state.
    a_nm : float
        Sphere radius [nm].
    T_K : float
        Temperature [K].
    wp_eV, eps_b, gamma_eV : float
        Drude parameters.
    Nr : int
        Radial grid points for G_l computation.
    lc_max : int
        Maximum Coulomb multipole.

    Returns
    -------
    gamma_matrix : float64[N, N]
        Rate matrix in 1/fs.
    """
    N = E_all.size
    gamma_matrix = np.zeros((N, N), dtype=np.float64)

    for j in prange(N):
        E_j = E_all[j]
        l_j = l_all[j]
        At2_j = At2_all[j]
        for i in range(N):
            if i == j:
                continue
            E_i = E_all[i]
            l_i = l_all[i]
            At2_i = At2_all[i]

            gamma_matrix[j, i] = _gamma_ee_pair(
                E_i, l_i, At2_i,
                E_j, l_j, At2_j,
                a_nm, T_K,
                wp_eV, eps_b, gamma_eV,
                Nr, lc_max
            )

    return gamma_matrix


# =============================================================================
# A6. e-e lifetimes  (SAG16 eqs 9, 10)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _ee_lifetimes_from_rates(
    gamma_matrix: np.ndarray,    # (N, N)
    E_all: np.ndarray,           # (N,)
    E_F: float,
    T_K: float,
) -> np.ndarray:
    """
    Compute e-e lifetimes from the rate matrix.

    For electrons (E_i > E_F):
        1/tau_i = sum_j (1 - f_j) * gamma[j, i]     (eq 9)

    For holes (E_i < E_F):
        1/tau_i = sum_j f_j * gamma[i, j]            (eq 10)
        (note: gamma[i,j] means scattering FROM j INTO i,
         but for holes, it's the rate of filling the hole)

    Parameters
    ----------
    gamma_matrix : float64[N, N]
        Rate matrix gamma[j, i] in 1/fs.
    E_all : float64[N]
        Bound state energies [eV].
    E_F : float
        Fermi energy [eV].
    T_K : float
        Temperature [K].

    Returns
    -------
    tau_ee : float64[N]
        e-e lifetimes in fs.
    """
    N = E_all.size
    tau_ee = np.empty(N, dtype=np.float64)

    fd = _fermi_dirac(E_all, E_F, T_K)

    for i in range(N):
        inv_tau = 0.0

        if E_all[i] >= E_F:
            # electron above Fermi level (eq 9)
            # 1/tau_i = sum_j (1 - f_j) * gamma_{j<-i}
            for j in range(N):
                if j == i:
                    continue
                inv_tau += (1.0 - fd[j]) * gamma_matrix[j, i]
        else:
            # hole below Fermi level (eq 10)
            # 1/tau_i = sum_j f_j * gamma_{i<-j}
            for j in range(N):
                if j == i:
                    continue
                inv_tau += fd[j] * gamma_matrix[i, j]

        if inv_tau > 1e-30:
            tau_ee[i] = 1.0 / inv_tau
        else:
            tau_ee[i] = 1e30  # effectively infinite lifetime

    return tau_ee


# =============================================================================
# A7. High-level wrappers (Python side)
# =============================================================================

def _flatten_states(e_state, a_nm: float):
    """
    Flatten List[QWLevelSet] into parallel arrays with dimensionless norms.

    The normalization is rescaled to avoid floating-point issues:
        |Ã|^2 = |A|^2 * a^3     (dimensionless, O(1))

    This absorbs the a^5 from the Coulomb prefactor:
        8*pi*k_C*a^5/hbar * |A_i|^2 * |A_j|^2
      = 8*pi*k_C/(hbar*a) * |Ã_i|^2 * |Ã_j|^2

    Returns
    -------
    E_all : float64[N]
    l_all : int64[N]
    At2_all : float64[N]   — dimensionless |Ã|^2
    """
    lmax = len(e_state)

    # count total states
    N = 0
    for l in range(lmax):
        N += e_state[l].Eb.size

    E_all = np.empty(N, dtype=np.float64)
    l_all = np.empty(N, dtype=np.int64)
    At2_all = np.empty(N, dtype=np.float64)

    a3 = a_nm ** 3   # scale factor

    idx = 0
    for l in range(lmax):
        n_l = e_state[l].Eb.size
        E_all[idx:idx+n_l] = e_state[l].Eb.real
        l_all[idx:idx+n_l] = l
        A_l = e_state[l].A
        for n in range(n_l):
            At2_all[idx + n] = (A_l[n] * A_l[n].conj()).real * a3
        idx += n_l

    return E_all, l_all, At2_all


def ee_scattering_rates(
    a_nm: float,
    e_state,
    T_K: float = 300.0,
    wp_eV: float = 9.07,
    eps_b: float = 4.18,
    gamma_eV: float = 0.06,
    Nr: int = 128,
    lc_max: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the full e-e scattering rate matrix.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    e_state : list[QWLevelSet] or numba.typed.List
        Quantum-well bound states per l.
    T_K : float
        Electron temperature [K].
    wp_eV, eps_b, gamma_eV : float
        Drude dielectric parameters.
    Nr : int
        Radial grid points for G_l.
    lc_max : int
        Maximum Coulomb multipole. If 0, uses len(e_state)-1.

    Returns
    -------
    gamma_matrix : float64[N, N]
        Rate matrix gamma[j, i] in 1/fs.
    E_all : float64[N]
        Energies of all states [eV].
    l_all : int64[N]
        Angular momentum index of each state.
    """
    E_all, l_all, At2_all = _flatten_states(e_state, a_nm)

    if lc_max <= 0:
        lc_max = len(e_state) - 1
    if lc_max < 1:
        lc_max = 1

    gamma_matrix = _ee_rate_matrix_parallel(
        E_all, l_all, At2_all,
        a_nm, T_K,
        wp_eV, eps_b, gamma_eV,
        Nr, lc_max,
    )

    return gamma_matrix, E_all, l_all


def ee_lifetimes(
    a_nm: float,
    E_F: float,
    e_state,
    T_K: float = 300.0,
    wp_eV: float = 9.07,
    eps_b: float = 4.18,
    gamma_eV: float = 0.06,
    Nr: int = 128,
    lc_max: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute electron-electron scattering lifetimes for all bound states.

    Based on Saavedra et al., ACS Photonics 3, 1637 (2016), eqs 9-10, 14.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Fermi energy [eV].
    e_state : list[QWLevelSet] or numba.typed.List
        Quantum-well bound states per l.
    T_K : float
        Electron temperature [K].
    wp_eV, eps_b, gamma_eV : float
        Drude dielectric parameters (default: Silver).
    Nr : int
        Radial grid points for the screened Coulomb integral.
    lc_max : int
        Maximum Coulomb multipole. If 0, uses len(e_state)-1.

    Returns
    -------
    tau_ee : float64[N]
        e-e lifetimes [fs] for each state.
    E_all : float64[N]
        Energies [eV], sorted by state index.
    l_all : int64[N]
        Angular momentum for each state.
    """
    gamma_matrix, E_all, l_all = ee_scattering_rates(
        a_nm, e_state,
        T_K=T_K, wp_eV=wp_eV, eps_b=eps_b, gamma_eV=gamma_eV,
        Nr=Nr, lc_max=lc_max,
    )

    tau_ee = _ee_lifetimes_from_rates(gamma_matrix, E_all, E_F, T_K)

    return tau_ee, E_all, l_all


# #############################################################################
#
#  PART B  –  ELECTRON-PHONON SCATTERING
#
#  Reference: Saavedra, Asenjo-Garcia & Garcia de Abajo,
#             ACS Photonics 3, 1637 (2016), "Inelastic Relaxation" section.
#
#  The e-ph coupling is phenomenological and state-independent:
#
#    (dp_i/dt)_{e-ph} = -γ^{e-ph} (p_i - p_i^0)
#
#  where p_i^0 = f(E_i, T_lattice) is the equilibrium Fermi-Dirac at the
#  lattice temperature, and:
#
#    γ^{e-ph} = G / c_e(T_e)
#
#  G is the electron-lattice coupling coefficient:
#    G_Au ≈ 3.0 × 10^16  W m^{-3} K^{-1}
#    G_Ag ≈ 3.5 × 10^16  W m^{-3} K^{-1}
#    G_Cu ≈ 1.0 × 10^17  W m^{-3} K^{-1}
#
#  c_e(T) is the electronic specific heat computed from the QW states:
#    c_e(T) = (2/V) × ∂/∂T Σ_i E_i f_i(T)
#
#  For large particles this recovers the Sommerfeld linear law c_e = γ_S T,
#  but for small particles finite-size effects produce non-monotonic behavior.
#
# #############################################################################

# ── Unit conversion constants ──────────────────────────────────────────────
# G is given in SI:  W m^{-3} K^{-1} = J s^{-1} m^{-3} K^{-1}
# Our units:  eV, fs, nm, K
#   1 J   = 1 / 1.602176634e-19 eV
#   1 s   = 1e15 fs
#   1 m^3 = 1e27 nm^3
# => 1 W m^{-3} K^{-1} = (1/1.602176634e-19) / (1e15 × 1e27) eV fs^{-1} nm^{-3} K^{-1}
#                        = 6.241509074e-24  eV fs^{-1} nm^{-3} K^{-1}
_SI_TO_eVfsNm3K = 6.241509074e-24

# Material e-ph coupling constants  (SI → our units)
EPH_COUPLING = {
    'Silver': 3.5e16 * _SI_TO_eVfsNm3K,   # eV/(fs·nm³·K)
    'Gold':   3.0e16 * _SI_TO_eVfsNm3K,
    'Copper': 1.0e17 * _SI_TO_eVfsNm3K,
}


# =============================================================================
# B1. Electronic specific heat from QW states  (SAG16 eq in text)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _ce_from_states(
    E_all: np.ndarray,     # (N,)  bound-state energies [eV]
    deg_all: np.ndarray,   # (N,)  degeneracy 2(2l+1) per state
    E_F: float,            # Fermi energy [eV]
    T_K: float,            # electron temperature [K]
    V_nm3: float,          # sphere volume [nm³]
) -> float:
    """
    Electronic specific heat at constant particle number N.

    Uses the canonical-ensemble variance formula which is always >= 0:

        c_e = (1 / (V k_B T²)) × (K2 - K1²/K0)

    where:
        K0 = Σ_i g_i f_i(1-f_i)            (number susceptibility)
        K1 = Σ_i g_i E_i f_i(1-f_i)        (mixed susceptibility)
        K2 = Σ_i g_i E_i² f_i(1-f_i)       (energy susceptibility)

    This correctly accounts for the temperature-dependent chemical
    potential (dμ/dT) at fixed N, avoiding the negative c_e that
    the naive grand-canonical derivative produces for discrete spectra.

    Parameters
    ----------
    E_all : float64[N]
        Bound-state energies [eV].
    deg_all : float64[N]
        Degeneracy per (n,l) level: 2(2l+1).
    E_F : float
        Chemical potential / Fermi energy [eV].
    T_K : float
        Electron temperature [K].
    V_nm3 : float
        Nanoparticle volume [nm³].

    Returns
    -------
    c_e : float
        Electronic specific heat [eV/(nm³·K)].
    """
    k_B = 8.617333262e-5   # eV/K
    N = E_all.size

    if T_K < 1.0:
        T_K = 1.0  # guard against division by zero

    kT = k_B * T_K

    K0 = 0.0   # Σ g_i f(1-f)
    K1 = 0.0   # Σ g_i E_i f(1-f)
    K2 = 0.0   # Σ g_i E_i² f(1-f)

    for i in range(N):
        E_i = E_all[i]
        arg = (E_i - E_F) / kT
        # guard against overflow in exp
        if arg > 500.0:
            f_i = 0.0
        elif arg < -500.0:
            f_i = 1.0
        else:
            f_i = 1.0 / (np.exp(arg) + 1.0)

        ff = f_i * (1.0 - f_i)
        g_i = deg_all[i]
        K0 += g_i * ff
        K1 += g_i * E_i * ff
        K2 += g_i * E_i * E_i * ff

    # Canonical variance:  Var(E)|_N = K2 - K1²/K0
    if K0 < 1e-30:
        return 0.0

    variance = K2 - (K1 * K1) / K0

    # c_e = variance / (V k_B T²)
    c_e = variance / (V_nm3 * k_B * T_K * T_K)

    return c_e


# =============================================================================
# B2. e-ph relaxation rate  γ^{e-ph} = G / c_e(T)
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _eph_rate_scalar(G_eVfsNm3K: float, c_e: float) -> float:
    """
    Electron-phonon relaxation rate.

    Parameters
    ----------
    G_eVfsNm3K : float
        e-ph coupling constant [eV/(fs·nm³·K)].
    c_e : float
        Electronic specific heat [eV/(nm³·K)].

    Returns
    -------
    gamma_eph : float
        Relaxation rate [1/fs].
    """
    if c_e < 1e-30:
        return 0.0
    return G_eVfsNm3K / c_e


# =============================================================================
# B3. High-level wrappers  (Python side)
# =============================================================================

def _flatten_states_with_deg(e_state):
    """
    Flatten states into arrays including degeneracy g = 2(2l+1).

    Returns
    -------
    E_all : float64[N]
    l_all : int64[N]
    deg_all : float64[N]   — degeneracy per (n,l) level
    """
    lmax = len(e_state)
    N = sum(e_state[l].Eb.size for l in range(lmax))

    E_all = np.empty(N, dtype=np.float64)
    l_all = np.empty(N, dtype=np.int64)
    deg_all = np.empty(N, dtype=np.float64)

    idx = 0
    for l in range(lmax):
        n_l = e_state[l].Eb.size
        g_l = 2.0 * (2 * l + 1)   # spin × magnetic degeneracy
        E_all[idx:idx+n_l] = e_state[l].Eb.real
        l_all[idx:idx+n_l] = l
        deg_all[idx:idx+n_l] = g_l
        idx += n_l

    return E_all, l_all, deg_all


def electronic_specific_heat(
    a_nm: float,
    E_F: float,
    e_state,
    T_K: float = 300.0,
) -> float:
    """
    Electronic specific heat from the QW bound states.

    c_e(T) = (1/V) Σ_i g_i × E_i × (E_i - μ)/(k_B T²) × f_i(1-f_i)

    For large particles this recovers the Sommerfeld linear law c_e ≈ γ_S T.
    For small particles, finite-size quantization produces deviations.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Chemical potential [eV].
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    T_K : float
        Electron temperature [K].

    Returns
    -------
    c_e : float
        Electronic specific heat [eV/(nm³·K)].
    """
    E_all, _, deg_all = _flatten_states_with_deg(e_state)
    V = (4.0 / 3.0) * np.pi * a_nm**3
    return _ce_from_states(E_all, deg_all, E_F, T_K, V)


def eph_rate(
    a_nm: float,
    E_F: float,
    e_state,
    T_K: float = 300.0,
    G_SI: float = 3.5e16,
) -> float:
    """
    Electron-phonon relaxation rate  γ^{e-ph} = G / c_e(T_e).

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Chemical potential [eV].
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    T_K : float
        Electron temperature [K].
    G_SI : float
        Electron-lattice coupling constant [W/(m³·K)].
        Default: 3.5×10^16 (Silver).

    Returns
    -------
    gamma_eph : float
        Relaxation rate [1/fs].
    """
    G_our = G_SI * _SI_TO_eVfsNm3K
    c_e = electronic_specific_heat(a_nm, E_F, e_state, T_K)
    return _eph_rate_scalar(G_our, c_e)


def eph_lifetime(
    a_nm: float,
    E_F: float,
    e_state,
    T_K: float = 300.0,
    G_SI: float = 3.5e16,
) -> float:
    """
    Electron-phonon relaxation lifetime  τ^{e-ph} = 1/γ^{e-ph} [fs].

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Chemical potential [eV].
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    T_K : float
        Electron temperature [K].
    G_SI : float
        Electron-lattice coupling constant [W/(m³·K)].
        Default: 3.5×10^16 (Silver).

    Returns
    -------
    tau_eph : float
        Relaxation lifetime [fs].
    """
    gamma = eph_rate(a_nm, E_F, e_state, T_K, G_SI)
    if gamma < 1e-30:
        return 1e30
    return 1.0 / gamma


def eph_rate_vs_temperature(
    a_nm: float,
    E_F: float,
    e_state,
    T_array_K: np.ndarray,
    G_SI: float = 3.5e16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute γ^{e-ph}(T) and c_e(T) over a temperature array.

    Useful for plotting the temperature dependence and comparing
    with the Sommerfeld (linear c_e) approximation.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Chemical potential [eV].
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    T_array_K : ndarray
        Temperature values [K].
    G_SI : float
        Electron-lattice coupling constant [W/(m³·K)].

    Returns
    -------
    gamma_eph : float64[N_T]
        Relaxation rate [1/fs] at each temperature.
    c_e : float64[N_T]
        Electronic specific heat [eV/(nm³·K)] at each temperature.
    """
    T_array_K = np.asarray(T_array_K, dtype=np.float64)
    G_our = G_SI * _SI_TO_eVfsNm3K

    E_all, _, deg_all = _flatten_states_with_deg(e_state)
    V = (4.0 / 3.0) * np.pi * a_nm**3

    N_T = T_array_K.size
    gamma_out = np.empty(N_T, dtype=np.float64)
    ce_out = np.empty(N_T, dtype=np.float64)

    for i in range(N_T):
        ce_i = _ce_from_states(E_all, deg_all, E_F, T_array_K[i], V)
        ce_out[i] = ce_i
        gamma_out[i] = _eph_rate_scalar(G_our, ce_i)

    return gamma_out, ce_out


# #############################################################################
#
#  PART C  –  EFFECTIVE LIFETIME  (e-e + e-ph combined)
#
#  Reference: Saavedra et al. ACS Photonics 3, 1637 (2016)
#
#  The total inelastic lifetime of state i is:
#
#    1/τ_eff(i)  =  1/τ_ee(i)  +  1/τ_eph
#
#  where τ_ee(i) is state-dependent (from Part A) and τ_eph is a single
#  scalar for all states (from Part B).  This combined lifetime replaces
#  the phenomenological τ used as a free parameter in hot_e_dist().
#
# #############################################################################


def effective_lifetime(
    a_nm: float,
    E_F: float,
    e_state,
    T_K: float = 300.0,
    wp_eV: float = 3.81,
    eps_b: float = 5.0,
    gamma_eV: float = 0.07,
    G_SI: float = 3.5e16,
    Nr: int = 128,
    lc_max: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Combined effective inelastic lifetime per state:  1/τ_eff = 1/τ_ee + 1/τ_eph.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Fermi energy [eV].
    e_state : list[QWLevelSet]
        Quantum-well states.
    T_K : float
        Electron temperature [K].
    wp_eV, eps_b, gamma_eV : float
        Drude parameters for the screened Coulomb interaction.
    G_SI : float
        Electron-phonon coupling constant [W/(m³·K)].
    Nr : int
        Radial grid points for the screened Coulomb Green's function.
    lc_max : int
        Maximum Coulomb multipole order (0 = auto).

    Returns
    -------
    tau_eff : float64[N]
        Effective lifetime per state [fs], energy-sorted order.
    E_sorted : float64[N]
        State energies [eV], sorted ascending.
    l_sorted : int64[N]
        Angular momentum quantum numbers, energy-sorted order.
    tau_eph : float
        Scalar e-ph lifetime [fs] (same for all states).
    """
    # --- e-e lifetimes (l-ordered) ---
    tau_ee, E_all, l_all = ee_lifetimes(
        a_nm, E_F, e_state,
        T_K=T_K, wp_eV=wp_eV, eps_b=eps_b, gamma_eV=gamma_eV,
        Nr=Nr, lc_max=lc_max,
    )

    # --- e-ph lifetime (scalar) ---
    tau_eph = eph_lifetime(a_nm, E_F, e_state, T_K=T_K, G_SI=G_SI)

    # --- combine: 1/τ_eff = 1/τ_ee + 1/τ_eph ---
    # States with infinite τ_ee (Pauli-blocked) contribute only τ_eph
    tau_ee_safe = np.where(np.isfinite(tau_ee) & (tau_ee > 0), tau_ee, 1e30)
    tau_eff = 1.0 / (1.0 / tau_ee_safe + 1.0 / tau_eph)

    # --- sort by energy (to match hot_e_dist output ordering) ---
    sort_idx = np.argsort(E_all)
    return tau_eff[sort_idx], E_all[sort_idx], l_all[sort_idx], tau_eph


def hot_e_dist_resolved(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_eff_sorted: np.ndarray,
    e_state,
    X_lm: np.ndarray,
    Pabs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Hot-carrier generation with state-resolved effective lifetimes.

    Each electronic state carries its own Lorentzian broadening
    γ_i = ħ/τ_eff[i].  For the electron (hole) channel the broadening
    is set by the lifetime of the initially occupied state:

      denom_e[f, i] = (ħω - E_f + E_i)² + (ħ/τ_eff[i])²   [electron, initial = i]
      denom_h[f, i] = (ħω - E_i + E_f)² + (ħ/τ_eff[f])²   [hole,     initial = f]

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    hv_eV : float
        Photon energy [eV].
    E_F : float
        Fermi energy [eV].
    tau_eff_sorted : float64[N]
        Effective lifetime per state [fs] in energy-sorted order.
        Obtain from :func:`effective_lifetime`.
    e_state : list[QWLevelSet]
        Quantum-well bound states.
    X_lm : ndarray
        BCM multipole coefficients at the photon wavelength.
    Pabs : float
        Absorbed power [eV/fs] (used for normalization).

    Returns
    -------
    Te, Th : float64[N]
        Hot electron/hole generation rate density [1/(fs·nm³)].
    Te_raw, Th_raw : float64[N]
        Unnormalized generation rates (before volume division) [1/fs].
    Mfi2_sorted : float64[N, N]
        Optical transition matrix |M_{fi}|² in energy-sorted order.
    E_sorted : float64[N]
        State energies [eV], sorted ascending.
    S : float
        Normalisation factor  Pabs / P_diss.
    Pabs_out : float
        Absorbed power [eV/fs] (echo of input).
    P_diss : float
        Computed dissipated power [eV/fs] before normalisation.
    """
    tau_eff_sorted = np.asarray(tau_eff_sorted, dtype=np.float64)

    # ── Step 1: get Mfi² matrix from the existing Numba kernel ──────────────
    # We call hot_e_dist with a dummy tau to extract Mfi2_sorted and E_sorted.
    # The matrix elements do NOT depend on τ — only the Lorentzian does.
    tau_dummy = np.array([500.0])
    _, _, _, _, Mfi2_sorted, E_sorted, _, _, _ = hot_e_dist(
        a_nm, hv_eV, E_F, tau_dummy, e_state, X_lm, Pabs
    )

    N = len(E_sorted)

    # ── Step 2: Fermi-Dirac occupations ─────────────────────────────────────
    fd = _fermi_dirac(E_sorted, E_F)
    fd_i = fd[np.newaxis, :]   # (1, N) — occupations indexed by initial state i
    fd_f = fd[:, np.newaxis]   # (N, 1) — occupations indexed by final/hole state f

    # ── Step 3: per-state Lorentzian widths  γ = ħ/τ ────────────────────────
    gamma_all = hbar / tau_eff_sorted          # shape (N,)
    gamma_col = gamma_all[np.newaxis, :]       # (1, N) — broadening from state i
    gamma_row = gamma_all[:, np.newaxis]       # (N, 1) — broadening from state f

    EE_f = E_sorted[:, np.newaxis]             # (N, 1)
    EE_i = E_sorted[np.newaxis, :]             # (1, N)

    # Denominators: each channel uses the lifetime of its *initial* occupied state
    denom_e = (hv_eV - EE_f + EE_i) ** 2 + gamma_col ** 2   # electron: initial = i
    denom_h = (hv_eV - EE_i + EE_f) ** 2 + gamma_row ** 2   # hole:     initial = f

    # ── Step 4: transition rate matrices ────────────────────────────────────
    # Prefactor 4γ/ħ = 4/τ (per-column for electrons, per-row for holes)
    pf_e = 4.0 * gamma_col / hbar    # (1, N)
    pf_h = 4.0 * gamma_row / hbar    # (N, 1)

    TTe = pf_e * fd_i * (1.0 - fd_f) * (
        Mfi2_sorted / denom_e + Mfi2_sorted.T / denom_h
    )
    TTh = pf_h * fd_f * (1.0 - fd_i) * (
        Mfi2_sorted / denom_h + Mfi2_sorted.T / denom_e
    )

    Te_raw = TTe.sum(axis=1)   # sum over initial states i
    Th_raw = TTh.sum(axis=1)

    # ── Step 5: normalise by absorbed power ─────────────────────────────────
    P_diss = 0.0
    for f in range(N):
        for i in range(N):
            dE = E_sorted[f] - E_sorted[i]
            if dE > 0.0:
                P_diss += dE * TTe[f, i]

    if P_diss <= 0.0 or not np.isfinite(P_diss):
        P_diss = 1e-300

    S = Pabs / P_diss
    Vol = (4.0 / 3.0) * np.pi * a_nm ** 3

    Te = S * Te_raw / Vol
    Th = S * Th_raw / Vol
    Te_raw_out = S * Te_raw
    Th_raw_out = S * Th_raw

    return Te, Th, Te_raw_out, Th_raw_out, Mfi2_sorted, E_sorted, S, Pabs, P_diss
