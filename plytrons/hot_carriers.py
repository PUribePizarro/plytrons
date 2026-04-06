from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as _NList

# ── Project‑specific helpers ───────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, nb_meshgrid, Wigner3j
from plytrons.quantum_well import js_real, ke, QWLevelSet

__all__ = ["hot_e_dist", "hot_e_cdf_per_photon", "count_eh_generated",
           "generation_efficiency"]

# =============================================================================
# 1. Low‑level utilities
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _fermi_dirac(E: np.ndarray, E_F: float, T: float = 0.0001) -> np.ndarray:
    """Vectorised Fermi–Dirac occupation *f(E)* (Numba‑accelerated)."""
    k_B = 8.617333262e-5  # eV K⁻¹
    return 1.0 / (np.exp((E - E_F) / (k_B * T)) + 1.0)

@nb.njit(cache=True)
def idx_to_lm(k: int) -> tuple[int, int]:
    """
    Constant-time conversion from array position `k` → (l, m).

    Parameters
    ----------
    k : int
        Position in the list (Python 0-based by default).
    one_based : bool, optional
        Set to True if your “cardinal index” starts at 1 instead of 0.

    Returns
    -------
    l, m : tuple[int, int]
    """
    if k < 0:
        raise IndexError("index must be ≥ 0")

    # ---- 1. find l ---------------------------------------------------------
    # total items up to and including l is Nl = l(l + 2)
    # minimal l with Nl > k  ⇒  l = ⌊√(k + 1) − 1⌋ + 1
    l = np.floor(np.sqrt(k + 1) - 1) + 1         # integer math, O(1)

    # ---- 2. local offset → m ----------------------------------------------
    offset = k - ((l - 1) * (l + 1))   # k minus items in previous blocks
    m = offset - l                      # map 0…2l  →  −l…l
    return l, m

@nb.njit(cache=True)
def lm_to_idx(l: int, m: int) -> int:
    """
    Constant-time conversion from (l, m) → array position k (0-based).

    Parameters
    ----------
    l : int
        Degree, must be >= 1.
    m : int
        Order, must satisfy -l <= m <= l.

    Returns
    -------
    k : int
        Position in the flattened (l,m) list (0-based).

    Notes
    -----
    Items are grouped by l, with sizes 2l+1. The number of items before
    block l is (l-1)(l+1). Inside the block, m maps to offset m + l.
    """
    if l < 1:
        raise IndexError("l must be ≥ 1")
    if m < -l or m > l:
        raise IndexError("m must satisfy −l ≤ m ≤ l")

    # items before this l-block + offset within block
    k = (l - 1) * (l + 1) + (m + l)
    return k

# -----------------------------------------------------------------------------
# 2. Single‑multipole transition matrix M_fi  (serial inside)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, parallel = False)
def _M_transition_squared(
    lf: int,
    li: int,
    a_nm: float,
    X_lm: np.ndarray,     # complex128[:]
    state_f: QWLevelSet,
    state_i: QWLevelSet,
) -> np.ndarray:
    """Compute *Mᶠᵢ* for a given pair of quantum numbers (l_f, m_f) ← (l_i, m_i)."""
    
    # get parameters of final state
    Ef, Af = state_f.Eb.real.astype(np.float64), state_f.A
    n_f = Ef.size

    # get parameters of initial state
    Ei, Ai = state_i.Eb.real.astype(np.float64), state_i.A
    n_i = Ei.size

    # |Af*Ai|^2 as an outer product -> (n_f, n_i)
    Af2 = (Af * Af.conj()).real
    Ai2 = (Ai * Ai.conj()).real
    AA_abs2 = Af2[:, None] * Ai2[None, :]

    # ----- radial grid & Bessels -------------------------------------------
    Nr = 128
    r = np.linspace(0.0, a_nm, Nr)      # (Nr,)
    rr = r[:, None]                          # (Nr, 1) for broadcasting

    # Bessel columns: A=(Nr,n_f), B=(Nr,n_i)
    j_lf = js_real(lf, ke(Ef[None, :]) * rr)       # j_lf(k_f r)
    j_li = js_real(li, ke(Ei[None, :]) * rr)       # j_li(k_i r)

    # trapezoid weights along r (Numba-safe)
    dr = 0.0 if Nr < 2 else (r[1] - r[0])
    w  = np.full(Nr, dr, dtype=np.float64)
    if Nr >= 1:
        w[0] *= 0.5
        w[-1] *= 0.5

    # will accumulate the real, positive squared amplitudes
    Mfi_2 = np.zeros((n_f, n_i), dtype=np.float64)

    # max l present in X_lm
    le_max = idx_to_lm(X_lm.size - 1)[0]

    for le in range(1, int(le_max) + 1):

        # triangle rule: |lf - li| <= le <= lf + li
        if le < abs(li - lf) or le > li + lf:
            continue

        # even-sum rule: lf + le + li must be even
        if ((lf + le + li) & 1) == 1:
            continue

       # ---------- Integration along solid angle ---------------------
        # power in field multipole le: P_le = sum_m |X_{le m}|^2
        idx0 = lm_to_idx(le, -le)
        idx1 = lm_to_idx(le,  le) + 1
        Xl = X_lm[idx0:idx1]

        # Numba-friendly real power sum
        X_lm_sum = 0.0
        for x_lm in Xl:
            X_lm_sum += x_lm.real * x_lm.real + x_lm.imag * x_lm.imag

        if X_lm_sum <= 1e-18:
            continue

        # angular factor from 3j orthogonality (sum over m_f, m_i)
        W = Wigner3j(lf, le, li, 0, 0, 0)
        Mfi_ang2 = ((2.0*lf + 1.0)*(2.0*li + 1.0)*(W*W)*X_lm_sum  # real\ 
                     / (4.0*np.pi))
                    
        # amplitude prefactor, squared (same physics as in _transition_M)
        # note: this is distinct from the orthogonality 1/(2le+1) that cancelled
        pref = (1.0/eps0) * np.sqrt(le / (a_nm**3)) / (2*le + 1)
        scale2 = (pref / (a_nm**(le - 1)))**2  # real

        # radial integrals for each Ef row
        # build weights r^(le+2)*w once per le
        rw = np.empty(Nr, dtype=np.float64)
        for ii in range(Nr):
            rw[ii] = w[ii] * (r[ii] ** (le + 2))

        # compute weighted B, then a single GEMM for all (f,i):
        # I = ∫ j_lf(k_f r) j_li(k_i r) r^(le+2) dr → (n_f, n_i)
        j_li_w = (rw[:, None]) * j_li                  # (Nr, n_i)
        I   = j_lf.T @ j_li_w                          # (n_f, n_i)

        # accumulate squared integral
        Mfi_2 += scale2 * Mfi_ang2 * (I * I)        # all real

    # include |Af*Ai|^2
    return Mfi_2 * AA_abs2

# =============================================================================
# 3. Parallel driver with full (l,m) summation
# =============================================================================

@nb.njit(fastmath=True, parallel=True)
def _hot_e_dist_parallel(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e: np.ndarray,
    e_state: List[QWLevelSet],
    X_lm: np.ndarray,
    Pabs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # --- flatten bound levels ---------------------------------------------
    lmax = len(e_state)
    l_range = np.zeros(lmax + 1, dtype=np.int64)
    for l in range(lmax):
        l_range[l + 1] = l_range[l] + e_state[l].Eb.size
    N = l_range[-1]

    E_all = np.empty(N, dtype=np.float64)
    for l in range(lmax):
        E_all[l_range[l]: l_range[l + 1]] = e_state[l].Eb.real

    # global transition matrices
    Mfi_2_all = np.zeros((N, N), dtype=np.float64)

    # outer parallelism over final l index ----------------------------------
    for lf in prange(lmax):
        state_lf = e_state[lf]
        lf_s, lf_e = l_range[lf], l_range[lf + 1]
        for li in range(lmax):
            state_li = e_state[li]
            li_s, li_e = l_range[li], l_range[li + 1]

            Mfi_2_block = _M_transition_squared(lf, li, a_nm, X_lm, state_lf, state_li)
            Mfi_2_all[lf_s:lf_e, li_s:li_e] = Mfi_2_block

# --- Golden‑rule probability matrices ----------------------------------
    EE_i, EE_f = nb_meshgrid(E_all, E_all)
    fd_i = _fermi_dirac(EE_i, E_F)
    fd_f = _fermi_dirac(EE_f, E_F)

    Te = np.zeros((len(tau_e), N), dtype=np.float64)
    Th = np.zeros((len(tau_e), N), dtype=np.float64)
    Te_raw_ = np.zeros((len(tau_e), N), dtype=np.float64)
    Th_raw_ = np.zeros((len(tau_e), N), dtype=np.float64)


    for i in range(len(tau_e)):

        tau_dx = tau_e[i]

        gamma_e = hbar / tau_dx  # eV    

        denom_e = (hv_eV - EE_f + EE_i)**2 + gamma_e**2
        denom_h = (hv_eV - EE_i + EE_f)**2 + gamma_e**2

        TTe = 4.0/tau_dx * fd_i * (1.0 - fd_f) * (
            Mfi_2_all/denom_e
              + Mfi_2_all.T/denom_h
            )
        
        TTh = 4.0/tau_dx * fd_f * (1.0 - fd_i) * (
            Mfi_2_all/denom_h
              + Mfi_2_all.T/denom_e
            )

        Te_raw = TTe.sum(axis=1)
        Th_raw = TTh.sum(axis=1)

        # --- Normalisation -----------------------------------------------------

        P_diss = 0.0
        for f in range(N):
            Ef = E_all[f]
            for i_ in range(N):
                dE = Ef - E_all[i_]
                if dE > 0.0:
                    P_diss += dE * (TTe[f, i_])

        if P_diss <= 0.0 or not np.isfinite(P_diss):
            P_diss = 1e-300

        S = Pabs / P_diss
        Vol = 4/3*np.pi*a_nm**3

        Te[i] = S * Te_raw/Vol
        Th[i] = S * Th_raw / Vol
        Te_raw_[i] = S * Te_raw
        Th_raw_[i] = S * Th_raw

    # ---- sort by energy (lowest → highest) and build a SORTED matrix ------
    order = np.argsort(E_all)
    E_sorted = E_all[order]

    # Numba-safe 2D reindex (explicit loops avoid fancy-index pitfalls)
    Mfi_2_sorted = np.empty_like(Mfi_2_all)
    for ii in range(N):
        i_old = order[ii]
        for jj in range(N):
            j_old = order[jj]
            Mfi_2_sorted[ii, jj] = Mfi_2_all[i_old, j_old]

    return Te, Th, Te_raw_, Th_raw_, Mfi_2_sorted, E_sorted, S, Pabs, P_diss


# =============================================================================
# 4. Thin wrapper
# =============================================================================

def hot_e_dist(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e_fs: np.ndarray,
    e_state,                # plain list OR numba.typed.List
    X_lm: np.ndarray,
    Pabs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """Parallel hot‑carrier generation with full (l,m) summation."""
    if not isinstance(e_state, _NList):
        tmp = _NList()
        for s in e_state:
            tmp.append(s)
        e_state = tmp
        
    return _hot_e_dist_parallel(a_nm, hv_eV, E_F, tau_e_fs, e_state, X_lm, Pabs)

# =============================================================================
# 5. Cumulative hot-electron yield per absorbed photon  N_e(ε)
#     N_e(ε) = (ħω / Pabs) * Σ_{E_f ≥ ε} Γ_e(E_f)
#     Here Γ_e(E_f) are the per-level generation rates in Te_raw_
# =============================================================================

def _flatten_energies_from_states(e_state: List[QWLevelSet]) -> np.ndarray:
    """Return the 1D array of all final-state energies E_f (eV) in the same
    order used internally when building transition blocks."""
    lmax = len(e_state)
    counts = [es.Eb.size for es in e_state]
    N = int(np.sum(np.array(counts)))
    E_all = np.empty(N, dtype=np.float64)
    s = 0
    for es in e_state:
        e = es.Eb.real.astype(np.float64)
        E_all[s:s+e.size] = e
        s += e.size
    return E_all

def hot_e_cdf_per_photon(
    Te_raw_: np.ndarray, e_state, hv_eV: float, Pabs: float, *,
    E_F: float = 0.0, eps_eval: np.ndarray | None = None, relative_to_EF: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # energies (in the same column order as Te_raw_)
    E_all = _flatten_energies_from_states(e_state)
    eps_ref = (E_all - E_F) if relative_to_EF else E_all

    # sort by ε and build right-cumulative sums
    idx = np.argsort(eps_ref)
    eps_sorted   = eps_ref[idx]
    Gamma_sorted = Te_raw_[:, idx]                  # (Ntau, N)
    Gamma_cum    = np.cumsum(Gamma_sorted[:, ::-1], axis=1)[:, ::-1]

    # row-wise totals (per τ)
    totals = Gamma_sorted.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0  # avoid /0 in pathological cases

    # This IS the “per absorbed photon” FoM (hv/Pabs cancels after rescaling)
    Ne_all_eps = Gamma_cum / totals                  # (Ntau, N)

    if eps_eval is None:
        return Ne_all_eps, eps_sorted
    else:
        eps_eval = np.asarray(eps_eval, dtype=np.float64)
        Ntau, M = Te_raw_.shape[0], eps_eval.size
        Ne = np.zeros((Ntau, eps_eval.size), dtype=np.float64)
        for m, th in enumerate(eps_eval):
            j = np.searchsorted(eps_sorted, th, side="left")
            Ne[:, m] = 0.0 if j >= eps_sorted.size else (Gamma_cum[:, j] / totals[:, 0])
        return Ne, eps_eval


def count_eh_generated(
    Te_raw_: np.ndarray,
    Th_raw_: np.ndarray,
    *,
    t_fs: float | np.ndarray,
    tau_index: int = 0,
    return_level_resolved: bool = False,
):
    """
    Count generated electrons and holes over a time window.

    Parameters
    ----------
    Te_raw_, Th_raw_ : (Ntau, N) arrays
        Level-resolved generation rates AFTER your scaling S:
            Te_raw_[i, f] = electron generation rate into final level f
            Th_raw_[i, f] = hole     generation rate into final level f
        Units: "per fs" if your internal time is fs (consistent with tau_e_fs).

    t_fs : float or array
        Time duration in femtoseconds over which to count carriers.
        If array, you get a time series of cumulative counts.

    tau_index : int
        Which row (i) to use (corresponding to tau_e_fs[i]).

    return_level_resolved : bool
        If True, also return per-level counts (same shape as N or (T,N)).

    Returns
    -------
    Ne, Nh : float or (T,) arrays
        Total number of electrons and holes generated in time t_fs.

    (optional) Ne_levels, Nh_levels : arrays
        Level-resolved counts. Shape (N,) if scalar t_fs, else (T, N).
    """
    Te_row = np.asarray(Te_raw_[tau_index], dtype=np.float64)
    Th_row = np.asarray(Th_raw_[tau_index], dtype=np.float64)

    t = np.asarray(t_fs, dtype=np.float64)

    # total rates (per fs)
    Ge = Te_row.sum()
    Gh = Th_row.sum()

    # totals (dimensionless counts)
    Ne = Ge * t
    Nh = Gh * t

    if not return_level_resolved:
        return Ne, Nh

    Ne_levels = t[..., None] * Te_row[None, ...] if t.ndim else t * Te_row
    Nh_levels = t[..., None] * Th_row[None, ...] if t.ndim else t * Th_row
    return Ne, Nh, Ne_levels, Nh_levels


# =============================================================================
# 6. Generation efficiency
# =============================================================================

def generation_efficiency(
    E_f: np.ndarray,
    Gamma_e: np.ndarray,
    omega_eV: float,
    Pabs: float,
) -> float:
    """
    Compute the hot-electron generation efficiency eta_g.

    eta_g is the fraction of absorbed photons that produce a hot electron
    above the Fermi level::

        eta_g = N_dot_e / (P_abs / hbar*omega)

    where ``N_dot_e = sum_f Gamma_e(E_f)`` is the total electron generation
    rate (sum of per-level rates over all final states *f*) and
    ``P_abs / hbar*omega`` is the absorbed-photon flux (photons per unit time).

    Parameters
    ----------
    E_f : ndarray, shape (N,)
        Final-state energies [eV].  Passed for caller identification and
        optional energy-resolved post-processing; not used in the
        summation itself.
    Gamma_e : ndarray, shape (N,)
        Per-level electron generation rate Gamma_e(E_f) [fs^-1].
        These are the values in ``Te_raw_[tau_index]`` returned by
        `hot_e_dist` after the absorbed-power normalisation.
    omega_eV : float
        Photon energy hbar*omega [eV].  Must be positive.
    Pabs : float
        Absorbed power [eV fs^-1].  Must be positive.

    Returns
    -------
    eta_g : float
        Dimensionless generation efficiency.  A value of 1 means every
        absorbed photon creates one hot electron; values less than 1
        indicate competing loss channels (e.g. intra-band absorption,
        momentum-forbidden transitions).

    Raises
    ------
    ValueError
        If ``Pabs`` or ``omega_eV`` is not strictly positive.

    Notes
    -----
    Units are consistent throughout:

    * ``N_dot_e = sum(Gamma_e)`` has units [fs^-1].
    * ``P_abs / hbar*omega`` has units [eV fs^-1] / [eV] = [fs^-1].
    * Their ratio eta_g is dimensionless.

    ``hbar*omega`` is simply ``omega_eV`` because ``Gamma_e`` and ``Pabs``
    are already expressed in the project's eV / fs unit system
    (``hbar = 0.6582 eV fs`` absorbed into the numerical values by
    `hot_e_dist`).

    Examples
    --------
    >>> Te, Th, Te_raw_, Th_raw_, Mfi2, E_sorted, S, Pabs, P_diss = hot_e_dist(...)
    >>> eta = generation_efficiency(E_sorted, Te_raw_[0], hv_eV, Pabs)
    >>> print(f"eta_g = {eta:.3f}")
    """
    E_f     = np.asarray(E_f,     dtype=np.float64)
    Gamma_e = np.asarray(Gamma_e, dtype=np.float64)

    if Pabs <= 0.0:
        raise ValueError(f"Pabs must be positive, got {Pabs!r}")
    if omega_eV <= 0.0:
        raise ValueError(f"omega_eV must be positive, got {omega_eV!r}")

    N_dot_e     = float(np.sum(Gamma_e))   # total electron generation rate [fs^-1]
    photon_rate = Pabs / omega_eV          # absorbed photon rate [fs^-1]

    return N_dot_e / photon_rate
