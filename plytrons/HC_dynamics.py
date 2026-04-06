# Hot Carriers Dynamics — Geometry & Optical-Excitation Analysis Layer
#
# Scope: quasi-static NP geometry sweeps and variable-geometry optical analysis.
# This module is intentionally limited to the BCM / optical-excitation layer:
#
#   bcm_sphere.py    ← EM solver  (BCM, X_lm coefficients)
#   hot_carriers.py  ← optical excitation  (Fermi golden rule, hot_e_dist)
#        ↑
#   HC_dynamics.py   ← THIS FILE
#
# Relaxation physics (electron-electron, electron-phonon) lives in
# relaxation.py and is NOT imported here — it is a separate model.
#
# Functions
# ---------
#   cluster_peak_energies      : group resonance peaks across a parameter sweep
#   gap_sweep                  : BCM solve across inter-sphere gap values
#   spatial_asymmetry_index    : per-sphere η_g asymmetry (0 = symmetric)
#   multipole_weight           : per-ℓ weight Σ_m |X_ℓm|² from BCM coefficients

import numpy as np

# ── Optical excitation core ────────────────────────────────────────────────────
from plytrons.hot_carriers import hot_e_dist

# ── plot_utils utilities used here ────────────────────────────────────────────
from plytrons.plot_utils import (
    find_dominant_peak,
    get_preset_geometry,
)

__all__ = [
    # Optical excitation (re-exported for caller convenience)
    "hot_e_dist",
    # Resonance tracking
    "cluster_peak_energies",
    "PEAK_ENERGY_TOL_eV",
    # Gap-sweep driver
    "MIN_GAP_TO_RADIUS_RATIO",
    "gap_sweep",
    # Cluster analysis
    "spatial_asymmetry_index",
    "multipole_weight",
]


# #############################################################################
#
#  PART E  –  SIMULATION VISUALIZATION UTILITIES
#
#  Functions for tracking resonances across parameter sweeps and generating
#  publication-quality GIFs combining absorption spectra and hot-carrier
#  distributions.
#
# #############################################################################

# ── Energy tolerance for resonance clustering ──────────────────────────────
PEAK_ENERGY_TOL_eV = 0.08

# ── Gap-geometry safety limit ───────────────────────────────────────────────
MIN_GAP_TO_RADIUS_RATIO = 0.05   # g_min = MIN_GAP_TO_RADIUS_RATIO * R


def cluster_peak_energies(energies_eV, tol_eV: float = PEAK_ENERGY_TOL_eV):
    """
    Cluster a list of peak energies (across gap/parameter sweep steps) into
    resonance groups separated by more than `tol_eV`.

    Parameters
    ----------
    energies_eV : array-like
        All detected peak energies [eV] across all sweep steps.
    tol_eV : float
        Minimum separation to consider two energies as distinct resonances.

    Returns
    -------
    clusters : list of dict
        Each dict has:
          'center_eV'   – mean energy of the cluster [eV]
          'members_idx' – original indices of energies_eV that belong here
    """
    E = np.array(energies_eV, float)
    if E.size == 0:
        return []
    order = np.argsort(E)
    E_sorted = E[order]
    clusters, start = [], 0
    for i in range(1, len(E_sorted) + 1):
        if i == len(E_sorted) or (E_sorted[i] - E_sorted[i - 1]) > tol_eV:
            members_sorted_idx = np.arange(start, i)
            members_orig_idx   = order[members_sorted_idx]
            center = E_sorted[start:i].mean()
            clusters.append({
                'center_eV':   float(center),
                'members_idx': members_orig_idx.tolist(),
            })
            start = i
    return clusters


def spatial_asymmetry_index(eta_g_per_sphere):
    """
    Compute the spatial asymmetry index A of hot-electron generation.

    .. math::

        A = \\frac{\\eta_{g,\\max} - \\eta_{g,\\min}}
                  {\\eta_{g,\\max} + \\eta_{g,\\min}}

    Parameters
    ----------
    eta_g_per_sphere : array-like or dict
        Per-sphere generation efficiencies eta_g.  For a dimer this is a
        length-2 sequence ``[eta_1, eta_2]``; for an N-sphere cluster,
        length N.  If a dict is passed (e.g. ``{label: eta_g}``), the
        values are used in iteration order.

    Returns
    -------
    A : float
        Asymmetry index in [0, 1]:

        * **A = 0** — perfectly symmetric: all spheres generate equally.
          Guaranteed for a homodimer under a symmetric illumination
          direction (e.g. E transverse, k through the gap midpoint).
        * **A = 1** — fully localised: generation is concentrated on one
          sphere (eta_g_min -> 0), i.e. one sphere acts as a pure
          absorber and the other as a near-field antenna.
        * **Intermediate values** indicate partial hot-spot localisation.
          In a homodimer swept from large to small gap, A typically rises
          as the bonding mode concentrates the near field at the gap
          facets of one sphere.

    Raises
    ------
    ValueError
        If fewer than 2 values are provided, or if eta_g_max + eta_g_min
        is zero (both efficiencies are identically zero).

    Examples
    --------
    >>> A = spatial_asymmetry_index([0.12, 0.08])
    >>> print(f"A = {A:.3f}")   # 0.200
    """
    if isinstance(eta_g_per_sphere, dict):
        vals = np.array(list(eta_g_per_sphere.values()), dtype=float)
    else:
        vals = np.asarray(eta_g_per_sphere, dtype=float).ravel()

    if vals.size < 2:
        raise ValueError(
            f"spatial_asymmetry_index requires at least 2 sphere values; "
            f"got {vals.size}."
        )

    eta_max = float(np.max(vals))
    eta_min = float(np.min(vals))
    denom   = eta_max + eta_min

    if denom == 0.0:
        raise ValueError(
            "eta_g_max + eta_g_min = 0; asymmetry index is undefined."
        )

    return (eta_max - eta_min) / denom


def multipole_weight(X_lm_coeffs):
    """
    Compute the per-multipole-order weight from BCM coefficients.

    For each angular momentum order ℓ, the weight is the sum of squared
    moduli over all magnetic quantum numbers m::

        W_ℓ = Σ_{m=-ℓ}^{ℓ} |X_{ℓm}|²

    Parameters
    ----------
    X_lm_coeffs : array-like, shape (n_coef,)
        Complex BCM surface-charge coefficients for one sphere at one
        frequency.  Accepted sources:

        * ``BCMObject.coef_at(lam_um)``  — shape (n_coef,)
        * ``coef[:, freq_idx]``          — one column of the sweep array

        Ordering follows the BCM convention in ``bcm_sphere.py``: modes
        are grouped by ℓ with m running from −ℓ to +ℓ, so the flat
        index for (ℓ, m) is::

            k(ℓ, m) = (ℓ - 1)(ℓ + 1) + (m + ℓ)

        and the total number of modes is ``n_coef = lmax*(lmax + 2)``.

    Returns
    -------
    weights : dict
        ``{ℓ: W_ℓ}`` for ℓ = 1, 2, …, lmax (int keys, float values).

    Raises
    ------
    ValueError
        If ``len(X_lm_coeffs)`` does not equal ``lmax*(lmax+2)`` for
        any integer lmax >= 1.

    Notes
    -----
    Physical interpretation:

    * For an isolated small sphere, ℓ = 1 (dipole) dominates.
    * As the inter-sphere gap narrows, the strongly inhomogeneous
      near-field drives higher multipoles: W_2 (quadrupole),
      W_3 (octupole), … grow relative to W_1.
    * Tracking ``multipole_weight`` across a gap sweep reveals the
      onset of multipolar coupling and signals the breakdown of the
      dipole approximation.

    Examples
    --------
    >>> X_lm = sphere.coef_at(lam_peak)      # (n_coef,) complex array
    >>> W = multipole_weight(X_lm)
    >>> for l, w in W.items():
    ...     print(f"  ℓ={l}: W={w:.4g}")
    """
    X = np.asarray(X_lm_coeffs, dtype=complex).ravel()
    n_coef = X.size

    # Invert  n_coef = lmax*(lmax+2) = (lmax+1)^2 - 1
    # => lmax + 1 = sqrt(n_coef + 1)  => lmax = round(sqrt(n_coef+1)) - 1
    lmax = int(round(np.sqrt(n_coef + 1.0))) - 1
    if lmax < 1 or lmax * (lmax + 2) != n_coef:
        raise ValueError(
            f"len(X_lm_coeffs) = {n_coef} is not consistent with any integer "
            f"lmax >= 1 (expected lmax*(lmax+2) for some lmax)."
        )

    weights = {}
    for l in range(1, lmax + 1):
        w = 0.0
        for m in range(-l, l + 1):
            k = (l - 1) * (l + 1) + (m + l)
            w += X[k].real ** 2 + X[k].imag ** 2
        weights[l] = w

    return weights


def gap_sweep(
    geometry_builder,
    gap_values_nm,
    omega_grid,
    params,
):
    """
    Sweep the inter-sphere gap, run a full BCM solve at each step, and
    locate the dominant absorption resonance.

    Parameters
    ----------
    geometry_builder : callable
        Factory ``g_nm -> list[BCMObject]``.  Called once per gap value
        with the gap in nm; must return a list of fully initialised
        ``BCMObject`` instances positioned for that gap.

        For the homodimer case use ``get_preset_geometry`` from
        ``plot_utils`` to obtain sphere centres, then wrap them in
        ``BCMObject``::

            from plytrons.plot_utils import get_preset_geometry
            import plytrons.bcm_sphere as bcm

            def dimer_builder(g):
                positions = get_preset_geometry('Dimer', D, g)
                return [bcm.BCMObject(label=f'Sp{i+1}', diameter=D,
                                      lmax=lmax, eps=eps_func,
                                      position=pos)
                        for i, pos in enumerate(positions)]

    gap_values_nm : array-like
        Gap values Delta [nm] to sweep.  Each value is validated before
        the BCM solve; a ``ValueError`` is raised if the gap is
        non-positive or below ``MIN_GAP_TO_RADIUS_RATIO * R``.
    omega_grid : ndarray
        Angular frequency axis [rad s^-1] at which the BCM is solved.
    params : dict
        Must contain:

        ``'efield'`` : EField
            Incident plane-wave object.
        ``'eps_h'`` : float
            Host medium permittivity.
        ``'sphere_diameter'`` : float
            Sphere diameter D [nm].  Used only for the gap guard;
            must match the diameter embedded in ``geometry_builder``.

    Returns
    -------
    dict with keys:

    ``'gap_nm'`` : ndarray, shape (Ngap,)
        Swept gap values [nm].
    ``'Qabs'`` : list of ndarray, each shape (Nomega,)
        Total cluster absorption efficiency at each gap step.
    ``'E_peak_eV'`` : ndarray, shape (Ngap,)
        Dominant resonance energy [eV] at each gap step
        (``nan`` when no peak is detected).
    ``'lam_peak_um'`` : ndarray, shape (Ngap,)
        Dominant resonance wavelength [um] at each gap step.
    ``'BCM_objects'`` : list of list
        ``BCMObject`` instances (with coefficients stored) per gap step.

    Raises
    ------
    ValueError
        If any gap value is non-positive or smaller than
        ``MIN_GAP_TO_RADIUS_RATIO * sphere_radius``.

    Notes
    -----
    The gap guard is applied **before** ``geometry_builder`` is called,
    so no BCM matrices are assembled for invalid geometries.

    Peak detection delegates to ``find_dominant_peak`` from
    ``plot_utils`` (prominence-based, same logic as ``label_peaks``).
    """
    import plytrons.bcm_sphere as bcm
    from scipy.constants import hbar as _hbar_si, eV as _eV_si
    from scipy.constants import physical_constants as _phys

    gap_values_nm = np.asarray(gap_values_nm, float)
    omega_grid    = np.asarray(omega_grid,    float)

    efield = params['efield']
    eps_h  = float(params['eps_h'])
    D      = float(params['sphere_diameter'])
    R      = D / 2.0

    # Derived axes
    lam_um = 2.0 * np.pi * 2.998e14 / omega_grid   # [um]  (c in um/s)
    E_eV   = omega_grid * _hbar_si / _eV_si          # [eV]

    # Incident intensity from efield (project units: eV fs^-1 nm^-2)
    _Z0_si = _phys["characteristic impedance of vacuum"][0]
    _Z0    = _Z0_si * _eV_si
    I0     = efield.E0 ** 2 / (2.0 * _Z0)

    Qabs_list   = []
    E_peaks     = []
    lam_peaks   = []
    bcm_lists   = []

    for g in gap_values_nm:
        g = float(g)

        # ── gap guard ────────────────────────────────────────────────────
        if g <= 0.0:
            raise ValueError(
                f"Gap must be positive; got g = {g:.6g} nm.  "
                "Spheres are touching or overlapping."
            )
        g_min = MIN_GAP_TO_RADIUS_RATIO * R
        if g < g_min:
            raise ValueError(
                f"Gap g = {g:.4f} nm is below the minimum allowed value "
                f"MIN_GAP_TO_RADIUS_RATIO * R = {MIN_GAP_TO_RADIUS_RATIO} "
                f"* {R:.3f} = {g_min:.4f} nm.  "
                "The BCM multipole expansion may not converge at this separation."
            )

        # ── geometry + matrices ──────────────────────────────────────────
        BCM_objects = geometry_builder(g)
        Np = len(BCM_objects)

        Gi = [bcm.Ginternal(obj) for obj in BCM_objects]
        G0 = [[bcm.Gexternal(BCM_objects[i], BCM_objects[j])
               for j in range(Np)] for i in range(Np)]
        Sv = [bcm.Efield_coupling(obj, efield) for obj in BCM_objects]

        # ── frequency sweep ──────────────────────────────────────────────
        n_coef   = BCM_objects[0].n_coef
        coef_all = [np.zeros((n_coef, len(omega_grid)), dtype=complex)
                    for _ in range(Np)]

        for il, wi in enumerate(omega_grid):
            c, _ = bcm.solve_BCM(wi, eps_h, BCM_objects, efield, Gi, G0, Sv)
            for sp in range(Np):
                coef_all[sp][:, il] = c[sp]

        for sp, obj in enumerate(BCM_objects):
            obj.set_coefficients(lam_um, coef_all[sp])

        # ── power and Q_abs ──────────────────────────────────────────────
        _, Pabs = bcm.EM_power(omega_grid, eps_h, Gi, G0, BCM_objects)
        Pabs_total = sum(Pabs[i] for i in range(Np))
        A_ref  = Np * np.pi * R ** 2          # total geometric cross-section [nm^2]
        Qabs   = Pabs_total / (I0 * A_ref)

        # ── dominant peak via plot_utils ─────────────────────────────────
        pk_idx, E_pk = find_dominant_peak(E_eV, Qabs)
        lam_pk = float(lam_um[pk_idx]) if pk_idx is not None else float('nan')
        E_pk   = E_pk if E_pk is not None else float('nan')

        Qabs_list.append(Qabs)
        E_peaks.append(E_pk)
        lam_peaks.append(lam_pk)
        bcm_lists.append(BCM_objects)

    return {
        'gap_nm':      gap_values_nm,
        'Qabs':        Qabs_list,
        'E_peak_eV':   np.array(E_peaks),
        'lam_peak_um': np.array(lam_peaks),
        'BCM_objects': bcm_lists,
    }


# GIF animation helpers (combined_absorption_hotcarriers_gif,
# static_multi_resonance_grid_gif) live in plot_utils.py — import
# them directly from there.
