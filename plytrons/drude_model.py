from typing import List, Tuple

import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as _NList   # ← NEW

# ── Project-specific helpers ───────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, nb_meshgrid, me, gaunt_coeff
from plytrons.quantum_well import js_real, ke, QWLevelSet, electrons_per_nanoparticle

# =============================================================================
# -----------------------------------------------------------------------------
# Drude bulk permittivity 
# -----------------------------------------------------------------------------
# ============================================================================= 


def eps_drude_bulk(omega_eV, wp_eV = 9.07, eps_b = 4.18, gamma0 = 0.06):
    """
    Drude bulk en las mismas unidades que eps_drude_PWA:
    omega_eV = ℏω en eV
    """
    omega = np.asarray(omega_eV, dtype=np.complex128)
    # ε(ω) = ε_b - ωp^2 / [ω (ω + iγ)]
    return eps_b - (wp_eV**2) / (omega * (omega + 1j * gamma0))

# =============================================================================
# -----------------------------------------------------------------------------
# Drude with correction from Nordlander 
# -----------------------------------------------------------------------------
# ============================================================================= 


def eps_drude_nano_nordlander(omega_eV, D_nm, wp_eV = 9.07, eps_b = 4.18, gamma0 = 0.06, vf = 1.39e6):
    """
    Drude con corrección de Nordlander para un nanopartícula de diámetro D_nm [nm],
    usando γ(R) = γ_bulk + ℏ v_F / R  (expresado todo en eV).

    omega_eV = ℏω en eV
    vf       = Fermi velocity [m/s] (default: Silver ~1.39e6 m/s)
    """
    omega = np.asarray(omega_eV, dtype=np.complex128)

    R_m = (D_nm * 1e-9) / 2.0   # radio en metros

    # ℏ en eV·s (constante física, no hbar de eV·fs)
    hbar_eVs = 6.582119569e-16

    gamma_size = hbar_eVs * vf / R_m      # eV
    gamma_eff  = gamma0 + gamma_size      # eV

    return eps_b - (wp_eV**2) / (omega * (omega + 1j * gamma_eff))


# =============================================================================
# -----------------------------------------------------------------------------
# García de Abajo PWA dipole transitions S_ij calculation
# -----------------------------------------------------------------------------
# ============================================================================= 


# =============================================================================
# 1. Low-level utilities
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _fermi_dirac(E: np.ndarray, E_F: float, T: float = 300.0) -> np.ndarray:
    """Vectorised Fermi–Dirac occupation *f(E)* (Numba-accelerated)."""
    k_B = 8.617333262e-5  # eV K⁻¹
    return 1.0 / (np.exp((E - E_F) / (k_B * T)) + 1.0)


@nb.njit(cache=True)
def idx_to_lm(k: int) -> tuple[int, int]:
    """
    Constant-time conversion from array position `k` → (l, m).

    Items are grouped by l, with sizes 2l+1. The number of items before
    block l is (l-1)(l+1). Inside the block, m maps to offset m + l.
    """
    if k < 0:
        raise IndexError("index must be ≥ 0")

    # total items up to and including l is Nl = l(l + 2)
    # minimal l with Nl > k  ⇒  l = ⌊√(k + 1) − 1⌋ + 1
    l = np.floor(np.sqrt(k + 1) - 1.0) + 1.0   # still float inside numba
    l = int(l)

    offset = k - ((l - 1) * (l + 1))   # k minus items in previous blocks
    m = offset - l                     # map 0…2l  →  −l…l
    return l, m


# =============================================================================
# 2. Single-multipole dipole matrix element <f|z|i>
# =============================================================================

@nb.njit(fastmath=True, cache=True)
def _transition_along_z(
    sf: Tuple[int, int],
    si: Tuple[int, int],
    a_nm: float,
    state_f: QWLevelSet,
    state_i: QWLevelSet,
) -> np.ndarray:
    """
    Dipole matrix <f|z|i> for a given pair (l_f, m_f) ← (l_i, m_i).

    Returns
    -------
    Mfi : complex128[:, :]
        Matrix of <f|z|i> for all (n_f, n_i), shape (n_f, n_i).
    """
    lf, mf = sf
    li, mi = si

    # ---- selection rules for z-polarisation -------------------------------
    # Δm = 0, Δl = ±1. If not satisfied → block is strictly zero.
    Ef = state_f.Eb.real.astype(np.float64)
    Ei = state_i.Eb.real.astype(np.float64)
    n_f = Ef.size
    n_i = Ei.size

    Mfi = np.zeros((n_f, n_i), dtype=np.complex128)

    if mf != mi:
        return Mfi
    if abs(lf - li) != 1:
        return Mfi

    Af = state_f.A
    Ai = state_i.A

    # ---- radial integrals: I_{fi} = ∫ j_lf j_li r^3 dr --------------------
    Nr = 128
    r = np.linspace(0.0, a_nm, Nr)
    rr = r[:, None]  # (Nr, 1)

    # j_lf(r, n_f)  → (Nr, n_f)
    j_lf = js_real(lf, ke(Ef[None, :]) * rr)
    # j_li(r, n_i)  → (Nr, n_i)
    j_li = js_real(li, ke(Ei[None, :]) * rr)

    # trapezoid weights along r
    dr = 0.0 if Nr < 2 else (r[1] - r[0])
    w = np.full(Nr, dr, dtype=np.float64)
    if Nr >= 1:
        w[0] *= 0.5
        w[-1] *= 0.5

    # r^3 * dr weights
    rw = np.empty(Nr, dtype=np.float64)
    for i in range(Nr):
        rw[i] = w[i] * (r[i] ** 3)

    # I = ∫ j_lf(k_f r) j_li(k_i r) r^3 dr → (n_f, n_i)
    j_li_w = rw[:, None] * j_li       # (Nr, n_i)
    I_rad = j_lf.T @ j_li_w           # (n_f, n_i)

    # ---- angular factor for z = r sqrt(4π/3) Y_1^0 -----------------------
    m = mi
    ga = gaunt_coeff(lf, 1, li, -m, 0, m)   # real
    sign = 1.0 if (m % 2 == 0) else -1.0
    I_ang = np.sqrt(4.0 * np.pi / 3.0) * sign * ga

    # ---- include normalisation constants ---------------------------------
    for nf in range(n_f):
        for ni in range(n_i):
            Mfi[nf, ni] = np.conj(Af[nf]) * Ai[ni] * I_ang * I_rad[nf, ni]

    return Mfi


# =============================================================================
# 3. Parallel driver: build S_ij for all pairs
# =============================================================================

@nb.njit(fastmath=True, parallel=True)
def Sij_parallel(
    a_nm: float,
    E_F: float,
    N_electrons: float,
    e_state: List[QWLevelSet],
) -> Tuple[np.ndarray, np.ndarray]:

    lmax = len(e_state)

    # --- flatten bound levels ---------------------------------------------
    l_range = np.zeros(lmax + 1, dtype=np.int64)
    for l in range(lmax):
        l_range[l + 1] = l_range[l] + e_state[l].Eb.size
    N = l_range[-1]

    E_all = np.empty(N, dtype=np.float64)
    for l in range(lmax):
        E_all[l_range[l]: l_range[l + 1]] = e_state[l].Eb.real

    # global |<f|z|i>|^2 matrix
    M2_all = np.zeros((N, N), dtype=np.float64)

    # ---- outer parallelism over final l index ----------------------------
    for lf in prange(lmax):
        state_lf = e_state[lf]
        n_f = state_lf.Eb.size
        lf_s, lf_e = l_range[lf], l_range[lf + 1]

        for li in range(lmax):
            state_li = e_state[li]
            n_i = state_li.Eb.size
            li_s, li_e = l_range[li], l_range[li + 1]

            # local block for this (lf, li)
            block = np.zeros((n_f, n_i), dtype=np.float64)

            # full m-sum (serial) – z-polarisation ⇒ mf = mi
            for m in range(-min(lf, li), min(lf, li) + 1):
                sf = (lf, m)
                si = (li, m)

                if abs(lf - li) == 1:
                
                    Mfi = _transition_along_z(sf, si, a_nm, state_lf, state_li)

                else:

                    Mfi = np.zeros((n_f, n_i), dtype=np.complex128)


                # accumulate |Mfi|^2
                for nf in range(n_f):
                    for ni in range(n_i):
                        val = Mfi[nf, ni]
                        block[nf, ni] += val.real * val.real + val.imag * val.imag

            # store block into global matrix
            for nf in range(n_f):
                for ni in range(n_i):
                    M2_all[lf_s + nf, li_s + ni] = block[nf, ni]

    # ---- build S_ij using Saavedra f-sum expression ----------------------
    EE_i, EE_f = nb_meshgrid(E_all, E_all)   # (N, N) each
    fd_i = _fermi_dirac(EE_i, E_F)
    fd_f = _fermi_dirac(EE_f, E_F)

    dE = EE_f - EE_i   # E_f - E_i

    # S_ij = 2 m_e / (ħ^2 N_e) * (E_f - E_i) * (f_i - f_f) * |<f|z|i>|^2
    pref = 2.0 * me / (hbar * hbar * N_electrons)
    S_mat = pref * dE * (fd_i - fd_f) * M2_all

    # ---- keep only upward transitions with positive S_ij -----------------
    # preallocate maximum possible size, then trim
    wij_tmp = np.empty(N * N, dtype=np.float64)
    S_tmp   = np.empty(N * N, dtype=np.float64)
    count = 0

    for nf in range(N):
        for ni in range(N):
            if EE_f[nf, ni] <= EE_i[nf, ni]:
                continue
            Sval = S_mat[nf, ni]
            if Sval <= 0.0:
                continue
            wij_tmp[count] = dE[nf, ni]      # Ω_ij = E_f - E_i  (eV)
            S_tmp[count]   = Sval
            count += 1

    wij = np.empty(count, dtype=np.float64)
    Sij = np.empty(count, dtype=np.float64)
    for k in range(count):
        wij[k] = wij_tmp[k]
        Sij[k] = S_tmp[k]

    return wij, Sij


# =============================================================================
# 4. Thin wrapper (Python side)
# =============================================================================

def Sij(
    a_nm: float,
    E_F: float,
    e_state,                # plain list OR numba.typed.List[QWLevelSet]
    N_atoms_uc: int = 4,
    alat_nm: float = 0.4086,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    High-level wrapper that returns all upward-transition frequencies and
    their corresponding S_ij.

    Parameters
    ----------
    a_nm : float
        Sphere radius [nm].
    E_F : float
        Fermi energy [eV].
    e_state : list[QWLevelSet] or numba.typed.List
        Quantum-well bound states per l.
    N_atoms_uc : int
        Atoms per unit cell (e.g. 4 for fcc).
    alat_nm : float
        Lattice constant [nm].

    Returns
    -------
    wij : ndarray
        Transition energies Ω_ij = E_f - E_i [eV].
    Sij : ndarray
        Dimensionless S_ij coefficients.
    Ne : float
        Total number of conduction electrons used in the normalisation.
    """
    # make sure e_state is a numba.typed.List for the jitted core
    if not isinstance(e_state, _NList):
        tmp = _NList()
        for s in e_state:
            tmp.append(s)
        e_state = tmp

    # total electrons in the nanoparticle (assuming 1 e⁻ / atom)
    D_nm = 2.0 * a_nm
    Ne = float(electrons_per_nanoparticle(D_nm, N_atoms_uc, alat_nm))

    wij, Sij_vals = Sij_parallel(a_nm, E_F, Ne, e_state)
    return wij, Sij_vals, Ne


# =============================================================================
# 5. Permittivity from PWA S_ij
# =============================================================================

from tqdm.auto import tqdm
import plytrons.quantum_well as qw

# def eps_PWA(a_nm, EF, e_states, omega_eV, eps_b=4, wp_eV=9, gamma_eV=0.06):
    
#     wij, Sij_vals, Ne = Sij(a_nm, EF, e_states)
#     omega_eV = np.asarray(omega_eV, float)
#     eps = eps_b + 0j*omega_eV
#     for W, S in zip(wij, Sij_vals):
#         eps += (wp_eV**2) * 2* S / (W**2 - omega_eV*(omega_eV + 1j*gamma_eV))
#     return eps


def eps_PWA(wij, Sij_vals,  omega_eV, wp_eV=9, eps_b=4, gamma_eV=0.06,
            show_progress=True):
    """
    Calcula ε_PWA(ω) con barra de progreso opcional.

    Parameters
    ----------
    omega_eV : array-like
        Energías ℏω en eV.
    wij : array-like
        Energías de transición Ω_ij en eV.
    Sij_vals : array-like
        Coeficientes S_ij correspondientes.
    show_progress : bool
        Si True, muestra tqdm en el loop sobre transiciones.
    """

    omega_eV = np.asarray(omega_eV, float)
    eps = eps_b + 0j * omega_eV

    iterable = zip(wij, Sij_vals)
    if show_progress:
        iterable = tqdm(iterable, total=len(wij), desc="Sumando transiciones PWA")

    for W, S in iterable:
        eps += (wp_eV**2) * 2 * S / (W**2 - omega_eV * (omega_eV + 1j * gamma_eV))

    return eps
