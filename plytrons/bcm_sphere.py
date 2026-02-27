"""
bcm_sphere.py
=============

Boundary-Charge Method (BCM) solver for *spherical* particles.
This standalone module is meant to be imported **from** projects that use
`plytrons.wigner3j` and `plytrons.math_utils`, but it is not itself a
sub-module of `plytrons`.  See the `EField`, `BCMObject`, and
`solve_BCM` docstrings for a quick-start guide.
"""

__all__ = [
    "EField", "BCMObject",
    "Ginternal", "Gexternal", "Efield_coupling", "solve_BCM",
    "EM_power", "MATERIAL_PARAMS", "MEDIA", "run_optical_response",
]

import numpy as np
from functools import lru_cache
import numba as nb
from math import factorial
from plytrons.wigner3j import clebsch_gordan
from plytrons.math_utils import em_sph_harm
from dataclasses import dataclass, field
from functools   import cached_property
from typing import Callable, Optional, Union

# -------------------------------------------------------------------------
# Dataclasses that describe the light source and each sphere
# -------------------------------------------------------------------------
@dataclass
class EField:
    """
    Plane-wave electric field.

    Parameters
    ----------
    E0 : float
        Field amplitude in the same units expected downstream (e.g., V nm⁻¹).
    k_hat : np.ndarray
        3-component unit vector for the propagation direction.
    e_hat : np.ndarray
        3-component unit vector for the polarisation.
    """
    E0: float
    k_hat: np.ndarray
    e_hat: np.ndarray

    def __post_init__(self):
        # Normalise direction vectors for safety
        self.k_hat = self.k_hat / np.linalg.norm(self.k_hat)
        self.e_hat = self.e_hat / np.linalg.norm(self.e_hat)


@dataclass
class BCMObject:
    """
    One spherical nanoparticle for the Boundary-Charge Method.

    Attributes
    ----------
    label : str
        Identifier used in log messages and plots.
    diameter : float
        Sphere diameter in nanometres.
    lmax : int
        Highest multipole order retained in the BCM expansion.
    eps : Callable[[float], complex]
        Function that returns the complex permittivity at a given wavelength.
    position : np.ndarray
        Centre position **r**₀ in nanometres.
    BCM_coef   : 2-D np.ndarray, optional
        Expansion coefficients  shape = (n_modes, n_λ)  *or* None until solved.
    lam_um     : 1-D np.ndarray, optional
        Wavelength grid associated with `BCM_coef`  (same n_λ)  *or* None.
    """
    label    : str
    diameter : float
    lmax     : int
    eps      : Callable[[float], complex]
    position : np.ndarray

    # ---------- internal storage  (underscore → “private”) -----------
    _coef   : Optional[np.ndarray] = field(default=None,  repr=False, init=False)
    _lam_um : Optional[np.ndarray] = field(default=None,  repr=False, init=False)

    # ----------------------------------------------------------------
    # Post-init check for *construction* invariants (no coef yet)
    # ----------------------------------------------------------------
    def __post_init__(self):
        if self.position.shape != (3,):
            raise ValueError("position must be a length-3 Cartesian vector")

    # --- new helper ----
    @cached_property
    def n_coef(self) -> int:
        """Total (l,m) coefficients kept for this sphere = lmax (lmax + 2)."""
        return self.lmax * (self.lmax + 2)

    # --- replacement for the old free function ----
    @cached_property
    def index_range(self) -> np.ndarray:
        """
        Indices that this particle occupies inside the *global* coefficient
        vector/matrix.  Always 0…n_coef-1 **before** any offsets are added
        by `solve_BCM`.
        """
        return np.arange(self.n_coef)

    # ----------------------------------------------------------------
    # Read-only *properties*  (no setter → external code can’t modify)
    # ----------------------------------------------------------------
    @property
    def BCM_coef(self) -> Union[np.ndarray, None]:
        return self._coef

    @property
    def lam_um(self) -> Union[np.ndarray, None]:
        return self._lam_um

    # ----------------------------------------------------------------
    # Single public mutator that guarantees BOTH arrays are consistent
    # ----------------------------------------------------------------
    def set_coefficients(self, lam_um: np.ndarray, coef: np.ndarray) -> None:
        """
        Store solver output in a single, atomic operation.

        Parameters
        ----------
        coef   : complex ndarray, shape (n_modes, n_λ)
        lam_um : float   ndarray, shape (n_λ,)

        Raises
        ------
        ValueError  if shapes mismatch.
        """
        if lam_um.ndim != 1:
            raise ValueError("lam_um must be 1-D")

        if coef.shape[-1] != lam_um.size:
            raise ValueError("coef last axis length must equal lam_um size")

        # All good – assign to the *private* slots
        self._coef   = coef.astype(np.complex128, copy=False)
        self._lam_um = lam_um.astype(np.float64,  copy=False)

    # ------------------------------------------------------------------
    # Helper: get expansion coefficients at arbitrary λ by interpolation
    # ------------------------------------------------------------------
    def coef_at(self, lam_query, *, extrapolate: bool = False) -> np.ndarray:
        """
        Interpolated BCM coefficients at the requested wavelength(s).

        Parameters
        ----------
        lam_query : float | np.ndarray
            Target wavelength(s) in um.  Scalar or 1-D array.
        extrapolate : bool, default False
            • False → raise ValueError if any λ is outside the stored range.  
            • True  → allow linear extrapolation using numpy.interp.

        Returns
        -------
        np.ndarray
            Complex array of shape (n_modes, Nq) where Nq is the number of
            query points (scalar → Nq = 1).
        """
        if self.BCM_coef is None or self.lam_um is None:
            raise RuntimeError("BCM_coef and lam_um must be set before "
                               "calling coef_at().")

        # Ensure query is 1-D NumPy array for uniform handling
        lam_q = np.atleast_1d(np.asarray(lam_query, dtype=np.float64))

        # Range check unless user explicitly wants extrapolation
        if (not extrapolate) and (
            (lam_q.min() < self.lam_um.min()) or (lam_q.max() > self.lam_um.max())
        ):
            raise ValueError("Query wavelength(s) outside stored lam_um range. "
                             "Set extrapolate=True to override.")

        n_modes = self.BCM_coef.shape[0]
        n_query = lam_q.size
        coef_q  = np.empty((n_modes, n_query), dtype=np.complex128)

        # Loop over modes (fast, small) and interpolate real & imag separately
        for m in range(n_modes):
            real_part = np.interp(lam_q, self.lam_um, self.BCM_coef[m].real)
            imag_part = np.interp(lam_q, self.lam_um, self.BCM_coef[m].imag)
            coef_q[m] = real_part + 1j * imag_part

        # If user supplied a scalar λ, squeeze trailing dimension for ergonomics
        if np.isscalar(lam_query):
            coef_q = coef_q[:, 0]

        return coef_q

@nb.njit
def BCM_basis_sphere(R, l, m, theta, phi):
    """
    The basis for spherical coordinates from boundary charge method (BCM).
    
    Input:
        R       : Sphere radius (nm)
        l       : Zenith index from spherical harmonics
        m       : Azimuth index from spherical harmonics
        theta   : Zenith angle
        phi     : Azimuth angle

    Output:
        Beta    : Function basis output
    """
    # Compute the basis function
    Beta = np.sqrt((2 * l + 1) / R**3) * em_sph_harm(m, l, theta, phi)  # spherical harmonics

    return Beta

@nb.njit
def BCM_proj_sphere(R, l, m, theta, phi):
    """
    Projection operator of sphere basis under the formalism of the
    boundary charge method (BCM).
    
    Input:
        R       : Sphere radius (nm)
        l       : Zenith index from spherical harmonics
        m       : Azimuth index from spherical harmonics
        theta   : Zenith angle (in radians)
        phi     : Azimuth angle (in radians)

    Output:
        varphi  : Function basis projector output
    """
    # Compute the projection operator
    varphi = R / (2 * l + 1) * np.conj(BCM_basis_sphere(R, l, m, theta, phi))  # spherical harmonics

    return varphi

def v_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def get_axis(vector):
    """Determine which standard axis a vector corresponds to"""
    # Check if the input is a 3-dimensional vector
    if len(vector) != 3:
        raise ValueError('Input must be a 3-dimensional vector')
    
    # Normalize to get 0 or 1 only
    vector = np.array(vector) / np.array(vector)
    vector[np.isnan(vector)] = 0
    
    # Check each component to determine the axis
    if np.array_equal(vector, [1, 0, 0]):
        return 'x'
    elif np.array_equal(vector, [0, 1, 0]):
        return 'y'
    elif np.array_equal(vector, [0, 0, 1]):
        return 'z'
    else:
        raise ValueError('Input vector does not correspond to a standard axis')

def Ginternal(BCM_object):
    """
    Calcula la matriz de interacción interna de una esfera utilizando el método
    de carga de frontera (BCM):
    
    Parámetros:
    -----------
    BCM_object : objeto BCMObject
        Objeto que contiene las propiedades de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    
    Retorna:
    --------
    Gin : ndarray
        Matriz de interacción interna
    
    Traducido de F. Ramirez 08/2019
    """
    # Asignar propiedades de la partícula
    lmax = BCM_object.lmax
    Ri = BCM_object.diameter / 2
    
    idx = 0
    idx_max = lmax * (lmax + 1) + (lmax + 1) - 1
    Gin = np.zeros((idx_max, idx_max), dtype = complex)
    
    for li in range(1, lmax + 1):
        for mi in range(-li, li + 1):
            Gin[idx, idx] = 1/2
            Gin[idx, idx] = Gin[idx, idx] * li / Ri**3  # factor adicional (conveniencia matemática)
            idx += 1
    
    return Gin

def Gexternal(BCM_object_i, BCM_object_j):
    """
    Calcula la matriz de interacción externa entre esferas utilizando el método
    de carga de frontera (BCM):
    
    Parámetros:
    -----------
    BCM_object_i : objeto BCMObject
        Objeto que contiene las propiedades de la primera partícula
        .label : Etiqueta de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    
    BCM_object_j : objeto BCMObject
        Objeto que contiene las propiedades de la segunda partícula
    
    Retorna:
    --------
    Gext : ndarray
        Matriz de interacción externa
    
    Traducido de F. Ramirez 08/2019
    """
    # Asignar propiedades de las partículas
    Label_i = BCM_object_i.label
    lmax_i = BCM_object_i.lmax
    Ri = BCM_object_i.diameter / 2
    xi = BCM_object_i.position

    Label_j = BCM_object_j.label
    lmax_j = BCM_object_j.lmax
    Rj = BCM_object_j.diameter / 2
    xj = BCM_object_j.position

    # -------------------------------------------------------------------------
    # Auto-interacción en espacio libre 1/2Ri x <Phi_i,Gii * beta_i>
    # -------------------------------------------------------------------------
    if Label_j == Label_i:
        idx = 0
        idx_max = lmax_i * (lmax_i + 1) + (lmax_i + 1) - 1
        Gext = np.zeros((idx_max, idx_max), dtype = complex)
        
        for li in range(1, lmax_i + 1):
            for mi in range(-li, li + 1):
                Gext[idx, idx] = 1 / 2 / (2 * li + 1)
                Gext[idx, idx] = Gext[idx, idx] * li / Ri**3  # factor adicional (conveniencia matemática)
                idx += 1
    
    # -------------------------------------------------------------------------
    # Interacción con un segundo objeto
    # -------------------------------------------------------------------------
    else:
        idx_max = lmax_i * (lmax_i + 1) + (lmax_i + 1) - 1
        jdx_max = lmax_j * (lmax_j + 1) + (lmax_j + 1) - 1
        Gext = np.zeros((idx_max, jdx_max), dtype = complex)
        
        idx = 0
        for li in range(1, lmax_i + 1):
            for mi in range(-li, li + 1):
                jdx = 0
                for lj in range(1, lmax_j + 1):
                    for mj in range(-lj, lj + 1):
                        Gext[idx, jdx] = Gcoupling_ij(li, mi, Ri, xi,
                                                 lj, mj, Rj, xj)
                        jdx += 1
                idx += 1
    
    return Gext

def Gcoupling_ij(li, mi, Ri, xi, lj, mj, Rj, xj):
    """
    Calcula el coeficiente de acoplamiento entre dos esferas.
    
    Parámetros:
    -----------
    li, mi : int
        Índices angulares de la primera esfera
    Ri : float
        Radio de la primera esfera
    xi : list o ndarray
        Posición de la primera esfera
    lj, mj : int
        Índices angulares de la segunda esfera
    Rj : float
        Radio de la segunda esfera
    xj : list o ndarray
        Posición de la segunda esfera
    
    Retorna:
    --------
    Gpq_ij : float o complex
        Coeficiente de acoplamiento
    """
    # -------------------------------------------------------------------------
    # Coordenadas del vector entre las dos esferas
    # -------------------------------------------------------------------------
    d_ij_x = xj[0] - xi[0]
    d_ij_y = xj[1] - xi[1]
    d_ij_z = xj[2] - xi[2]

    # -------------------------------------------------------------------------
    # Obtener módulo y ángulos del vector entre las dos esferas
    # -------------------------------------------------------------------------
    d_ij_mod = np.sqrt(d_ij_x**2 + d_ij_y**2 + d_ij_z**2)  # módulo del vector
    tt_ij = np.arctan2(np.sqrt(d_ij_x**2 + d_ij_y**2), d_ij_z)  # ángulo cenital
    pp_ij = np.arctan2(d_ij_y, d_ij_x)  # ángulo azimutal

    # -------------------------------------------------------------------------
    # Calcular la integral:
    #       Y(li,mi)(Omega_i)*Y(lj,mj)(Omega_ij)/r_ij^(lj+1) dOmega_q
    # -------------------------------------------------------------------------
    int_ij = ((-1)**(lj-mi)) * ((2*lj + 1)/(2*(li+lj) + 1)) \
            * np.sqrt(4*np.pi*factorial(2*(li+lj) + 1) / \
             (factorial(2*li + 1)*factorial(2*lj + 1))) \
            * clebsch_gordan(li, -mi, lj, mj, li+lj, mj-mi) \
            * em_sph_harm(mj-mi, li+lj, tt_ij, pp_ij) \
            * Ri**li / d_ij_mod**(li+lj+1)

    # -------------------------------------------------------------------------
    # Terminar el cálculo agregando factores adicionales
    # -------------------------------------------------------------------------

    # Estos factores son independientes de la base
    Gpq_ij = -(Rj/Ri) * Rj**(lj + 1) / (2*lj + 1) * int_ij  # base: Ylm(Omega)
    Gpq_ij = np.sqrt(lj/Rj**3) * np.sqrt(li/Ri**3) * Gpq_ij  # base: sqrt(l/R^3)Ylm(Omega)

    # Agregar factores para base específica
    Gpq_ij = np.sqrt(lj*li) * (Ri/Rj)**(3/2) * Gpq_ij
    
    return Gpq_ij

def trapz2(Z, x=None, y=None):
    """
    Compute the double integral of a 2D array using the trapezoidal method.
    
    Parameters:
    -----------
    Z : ndarray
        The 2D array to integrate.
    x : ndarray, optional
        The x coordinates. If None, assumes uniform spacing.
    y : ndarray, optional
        The y coordinates. If None, assumes uniform spacing.
    
    Returns:
    --------
    float
        The result of the double integration.
    
    Notes:
    ------
    This function implements the equivalent of MATLAB's trapz2 function.
    It first integrates along the columns (axis=0) and then along the rows (axis=1).
    """
    # First integration along columns (axis=0)
    if y is None:
        col_integral = np.trapz(Z, axis=0)
    else:
        col_integral = np.trapz(Z, y, axis=0)
    
    # Second integration along rows (axis=1) of the result from the first integration
    if x is None:
        result = np.trapz(col_integral)
    else:
        result = np.trapz(col_integral, x)
    
    return result

@lru_cache(maxsize=None)
def _prepare_grid(n_theta: int = 150, n_phi: int = 30):
    """Build and cache the (θ,φ) mesh plus helpers."""
    θ = np.linspace(0, np.pi,  n_theta)
    φ = np.linspace(0, 2*np.pi, n_phi)
    tt, pp = np.meshgrid(θ, φ)        # shape (nθ, nφ)

    sinθ   = np.sin(tt)
    n_hat  = np.stack((sinθ*np.cos(pp),               # x
                       sinθ*np.sin(pp),               # y
                       np.cos(tt)), axis=0)           # z  – shape (3,nθ,nφ)
    return θ, φ, tt, pp, sinθ, n_hat

def Ecoupling_matrix(lmax: int, Ri: float, e_hat: np.ndarray,
                     n_theta: int = 150, n_phi: int = 30) -> np.ndarray:
    """
    Vectorised replacement for Ecoupling_coef.
    Returns Xi for *all* (l,m) with 1 ≤ l ≤ lmax, -l ≤ m ≤ l.
    """
    θ, φ, tt, pp, sinθ, n_hat = _prepare_grid(n_theta, n_phi)

    # n·e  on the whole grid — broadcast e_hat over (nθ,nφ)
    ne_dot = (n_hat[0]*e_hat[0] +
              n_hat[1]*e_hat[1] +
              n_hat[2]*e_hat[2])                      # shape (nθ,nφ)

    # --- build flattened arrays of l and m ---
    li, mi = [], []
    for l in range(1, lmax+1):
        li.extend([l]*(2*l+1))     # repeats l, (2*l+1) times
        mi.extend(range(-l, l+1))
    li = np.asarray(li)
    mi = np.asarray(mi)

    # Broadcast (l,m) over the grid → Ylm shape (modes,nθ,nφ)
    Ylm = em_sph_harm(mi[:, None, None], li[:, None, None], tt, pp)

    # Integrand:  conj(Ylm) * (n·e)                       (broadcasted)
    integrand = np.conj(Ylm) * ne_dot[None, :, :]
    
    #  --- 2d trapezoidal rule (same as trapz2)  --- 
    dθ  = θ[1] - θ[0]
    dφ  = φ[1] - φ[0]

    # Create 1D trapezoidal weights for x and y
    wθ = np.ones_like(θ)
    wθ[0] = wθ[-1] = 0.5
    wφ = np.ones_like(φ)
    wφ[0] = wφ[-1] = 0.5

    # Create 2D weights by outer product
    W = np.outer(wφ, wθ)
    
    # Perform the double integral using the trapezoidal rule
    # Xi = ∫∫ integrand * sin(θ) dθ dφ
    Xi  = np.tensordot(integrand * sinθ[None, :, :],   # shape (modes,nθ,nφ)
                       W,                              # weights shape (nφ,nθ)
                       axes=([1,2],[0,1])) * dθ * dφ

    #  ---  Final radial prefactor √(l / R³)  --- 
    Xi *= np.sqrt(li / Ri**3)
    return Xi

def Efield_coupling(obj_i: BCMObject, Efield: EField,
                    n_theta: int = 150, n_phi: int = 30) -> np.ndarray:
    """Vectorised – no inner (l,m) loops."""
    eps0 = 0.055263493756           # vacuum permittivity (e / V·nm)
    Ri   = obj_i.diameter / 2
    Xi   = Ecoupling_matrix(obj_i.lmax, Ri, Efield.e_hat,
                            n_theta=n_theta, n_phi=n_phi)
    Si   = eps0 * Efield.E0 * Xi
    return Si

def solve_BCM(w, eps_h, BCM_objects, Efield, Gint, Gext, Sext):
    """
    Calcula los coeficientes de la expansión de densidad de carga superficial:
    
    Parámetros:
    -----------
    w : float
        Frecuencia (rad/s)
    eps_h : float
        Constante dieléctrica del medio
    BCM_objects : list
        Lista de objetos BCM con propiedades de las partículas
        .label : Etiqueta de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Función de constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    Efield : objeto ElectricField
        Propiedades de la fuente del campo E
        .E0 : Intensidad del campo E (W/m)
        .e_hat : Dirección del vector del campo E
        .k_hat : Dirección de la onda plana
    Gint : list
        Lista con matrices de interacción interna
    Gext : list of lists
        Lista de listas con matrices de interacción externa
    Sext : list
        Lista con coeficientes de acoplamiento de la fuente
    
    Retorna:
    --------
    X : list
        Lista de coeficientes de expansión para cada objeto
    Sw : list
        Lista de vectores de fuente para cada objeto
    
    Traducido de F. Ramirez 08/2019
    """
    # -------------------------------------------------------------------------
    # Definir constantes
    # -------------------------------------------------------------------------
    c0 = 299792458  # Velocidad de la luz (m/s)
    # Calcular vector de onda externo
    kh = w / c0 * np.sqrt(eps_h) * 1E-9  # vector de onda (1/nm)
    k_hat = Efield.k_hat
    E0 = Efield.E0
    lambda_value = 2 * np.pi * c0 / w * 1E6
    
    # -------------------------------------------------------------------------
    # Calcular sistema de matrices
    # -------------------------------------------------------------------------
    Np = len(BCM_objects)
    G_all_size = 0
    for in_idx in range(Np):
        # Calcular el tamaño total de la matriz de interacción
        G_all_size = G_all_size + Gint[in_idx].shape[0]
    
    G_all = np.zeros((G_all_size, G_all_size), dtype=complex)
    S_all = np.zeros(G_all_size, dtype=complex)
    
    idx_last = 0
    for in_idx in range(Np):
        # Calcular constante eta
        eps_i = BCM_objects[in_idx].eps(lambda_value)
        eta_i = (eps_i + eps_h) / (eps_i - eps_h)
        
        # Obtener el rango de índices de la matriz del objeto i
        i_range = idx_last + BCM_objects[in_idx].index_range
        
        # Almacenar matriz de interacción interna
        G_all[np.ix_(i_range, i_range)] = eta_i * Gint[in_idx]
        
        jdx_last = 0
        for jn_idx in range(Np):
            # Obtener el rango de índices de la matriz del objeto j
            j_range = jdx_last + BCM_objects[jn_idx].index_range
            
            # Almacenar matriz de interacción externa
            G_all[np.ix_(i_range, j_range)] = G_all[np.ix_(i_range, j_range)] - Gext[in_idx][jn_idx]
            
            # Actualizar índice
            jdx_last = j_range[-1]+1
        
        # Almacenar vector de interacción de campo E externo
        kh_ri = np.dot(k_hat, BCM_objects[in_idx].position)
        S_all[i_range] = Sext[in_idx] * np.exp(-1j * kh * kh_ri)
        
        # Actualizar índice
        idx_last = i_range[-1]+1
    
    # -------------------------------------------------------------------------
    # Resolver sistema lineal y almacenar resultados
    # -------------------------------------------------------------------------
    X_all_result = np.linalg.solve(G_all, S_all)
    X_all = X_all_result
    
    X = [None] * Np
    Sw = [None] * Np
    
    idx_last = 0
    for in_idx in range(Np):  # Using in_idx as 'in' is a reserved keyword in Python
        # Get the matrix index range from i-object
        i_range = idx_last + BCM_objects[in_idx].index_range
        
        # store results for each object - ensure i_range is properly used for indexing
        X[in_idx] = X_all[i_range]  # This should work since i_range is a numpy array
        Sw[in_idx] = S_all[i_range]
        
        # update index
        idx_last = i_range[-1]+1  # +1 to start at the next index
    
    return X, Sw

def EM_power(w, eps_h, Gint, Gext, BCM_objects):
    """
    Calcula la potencia EM dispersada y absorbida.
    
    Parámetros:
    -----------
    w : ndarray
        Frecuencia de cálculo (rad/s)
    eps_h : float
        Constante dieléctrica del medio
    Gint : list
        Lista con matrices de interacción interna
    Gext : list of lists
        Lista de listas con matrices de interacción externa
    BCM_objects : list
        Lista de objetos BCM con propiedades de las partículas
        .label      : Etiqueta de la partícula
        .diameter   : Diámetro de la partícula (nm)
        .lmax       : Número máximo de índice angular
        .eps        : Función de constante dieléctrica de la partícula
        .position   : Posición de la partícula (nm)
        .BCM_coef   : Coeficientes BCM calculados
    
    Retorna:
    --------
    Psca : list
        Lista de potencias dispersadas (W/umm^2)
    Pabs : list
        Lista de potencias absorbidas (W/umm^2)
    
    Traducido de F. Ramirez 08/2019
    """
    # -------------------------------------------------------------------------
    # Definir constantes
    # -------------------------------------------------------------------------
    eps0 = 0.055263493756                  # Permitividad del vacío (e/V-nm)
    c0 = 299792458                         # Velocidad de la luz (m/s)
    
    # -------------------------------------------------------------------------
    # Calcular potencia por objeto
    # -------------------------------------------------------------------------
    Np = len(BCM_objects)
    Pabs = [None] * Np
    Psca = [None] * Np
    
    for in_idx in range(Np):
        # Calcular constante eta
        R = BCM_objects[in_idx].diameter / 2
        
        Pabs[in_idx] = np.zeros(len(w))
        Psca[in_idx] = np.zeros(len(w))
        
        for iw in range(len(w)):
            kh = w[iw] / c0 * np.sqrt(eps_h) * 1E-9    # Vector de onda externo (1/nm)
            lambda_value = 2 * np.pi * c0 / w[iw] * 1E6  # Longitudes de onda
            
            eps_i = BCM_objects[in_idx].eps(lambda_value)
            eta_i = (eps_i + eps_h) / (eps_i - eps_h)
            
            # Calcular matriz de interacción interna
            G_i = eta_i * Gint[in_idx] - Gext[in_idx][in_idx]
            
            # Extraer coeficiente de expansión para la frecuencia requerida
            Xi = BCM_objects[in_idx].BCM_coef[:, iw]
            
            # Calcular potencia absorbida
            Pabs[in_idx][iw] = -np.real(w[iw] * R**3 / (2 * eps0) * 
                                       np.dot(np.conj(Xi).T, np.dot(asym(G_i), Xi)))
            
            # Calcular potencia dispersada
            Psca[in_idx][iw] = -np.real(w[iw] * (kh * R)**3 / (12 * np.pi * eps0) * 
                                       np.dot(np.conj(Xi).T, np.dot(asym(eta_i * G_i), Xi)))
    
    return Psca, Pabs

def asym(M):
    """
    Calcula la parte antisimétrica de una matriz.
    
    Parámetros:
    -----------
    M : ndarray
        Matriz de entrada
    
    Retorna:
    --------
    asymM : ndarray
        Parte antisimétrica de la matriz M
    """
    return 1/(2j) * (M - np.conj(M.T))


# ─────────────────────────────────────────────────────────────────────────────
# High-level workflow: build objects, solve BCM, compute power spectra
# ─────────────────────────────────────────────────────────────────────────────

# Material and media databases used by run_optical_response
MATERIAL_PARAMS = {
    'Silver':  {'EF': 5.53, 'wp': 9.07, 'vf': 1.39e6, 'gamma0': 0.060, 'eps_b': 4.18},
    'Gold':    {'EF': 5.53, 'wp': 9.03, 'vf': 1.40e6, 'gamma0': 0.070, 'eps_b': 9.84},
    'Copper':  {'EF': 7.0,  'wp': 8.8,  'vf': 1.57e6, 'gamma0': 0.027, 'eps_b': 1.0},
}

MEDIA = {
    'Vacuum / Air': 1.0,
    'Water':        1.77,
    'Glass (SiO₂)': 2.25,
    'TiO₂':         6.25,
}


def _make_eps_drude(mat_name, D, model, material_params, _cache={}):
    """Build a callable eps(lam_um) for a nanoparticle, with caching."""
    from scipy.constants import hbar, eV, speed_of_light
    import plytrons.quantum_well as qw
    from plytrons.drude_model import (
        Sij as compute_Sij, eps_PWA,
        eps_drude_nano_nordlander, eps_drude_bulk,
    )
    c0 = speed_of_light

    key = (mat_name, D, model)
    if key in _cache:
        return _cache[key]

    p = material_params[mat_name]

    if model == 'PWA':
        E_mat = qw.get_bound_states(D / 2)
        A_mat = qw.get_normalization(D / 2, E_mat)
        e_st  = qw.e_state_assembly(E_mat, A_mat)
        wij_i, Sij_i, _ = compute_Sij(D / 2, p['EF'], e_st)
        mask  = np.isfinite(wij_i) & np.isfinite(Sij_i) & (wij_i > 0) & (Sij_i > 0)
        wij_i = wij_i[mask]; Sij_i = Sij_i[mask]

        def eps(x, _wij=wij_i, _Sij=Sij_i, _wp=p['wp'], _eb=p['eps_b'], _gam=p['gamma0']):
            E_eV = (hbar / eV) * (2 * np.pi * c0 / (x * 1e-6))
            with np.errstate(divide='ignore', invalid='ignore'):
                val = eps_PWA(_wij, _Sij, E_eV, wp_eV=_wp, eps_b=_eb, gamma_eV=_gam, show_progress=False)
            if not np.isfinite(val):
                val = _eb - _wp**2 / (E_eV * (E_eV + 1j * _gam))
            return val

    elif model == 'Nordlander':
        def eps(x, _D=D, _wp=p['wp'], _eb=p['eps_b'], _gam=p['gamma0'], _vf=p['vf']):
            E_eV = (hbar / eV) * (2 * np.pi * c0 / (x * 1e-6))
            return eps_drude_nano_nordlander(E_eV, D_nm=_D, wp_eV=_wp, eps_b=_eb, gamma0=_gam, vf=_vf)

    elif model == 'Bulk':
        def eps(x, _wp=p['wp'], _eb=p['eps_b'], _gam=p['gamma0']):
            E_eV = (hbar / eV) * (2 * np.pi * c0 / (x * 1e-6))
            return eps_drude_bulk(E_eV, wp_eV=_wp, eps_b=_eb, gamma0=_gam)

    else:
        raise ValueError(f"Unknown model: {model!r}. Choose 'PWA', 'Nordlander', or 'Bulk'.")

    _cache[key] = eps
    return eps


def run_optical_response(config, *, lmax=10, n_points=500, pad_low=0.2, pad_high=0.2):
    """
    Run the full BCM optical-response workflow from a builder config dict.

    Parameters"""
    import warnings
    warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)
    warnings.filterwarnings("ignore", category=nb.NumbaWarning)
    warnings.filterwarnings("ignore", message=".*NumbaTypeSafetyWarning.*")
    warnings.filterwarnings("ignore", message=".*Cannot cache.*")
    """
    ----------
    config : dict
        Must contain keys: 'positions', 'diameters', 'materials',
        'k_vec', 'e_vec', 'eps_h', 'model'.
    lmax : int
        Maximum multipole order.
    n_points : int
        Number of frequency points.
    pad_low, pad_high : float
        Fractional padding around the auto-detected resonance range.

    Returns
    -------
    dict with keys:
        BCM_objects, efield, w, lam_um, E_eV, eps_h, I0, Z0,
        Psca, Pabs, Qsca_total, Qabs_total, Qext_total,
        peak_idx_total, peak_idx_per_particle, outdir
    """
    from scipy.constants import hbar, eV, speed_of_light, physical_constants
    from scipy.signal import find_peaks
    from tqdm.auto import tqdm, trange
    from plytrons.plot_utils import make_results_folder

    c0 = speed_of_light
    Z0_val, *_ = physical_constants["characteristic impedance of vacuum"]
    Z0_val = Z0_val * eV

    # ── Medium ──
    eps_h = MEDIA[config['eps_h']]
    print(f"Surrounding medium: {config['eps_h']}  (εₕ = {eps_h})")

    # ── Auto frequency range ──
    resonances = []
    for mat in config['materials']:
        p = MATERIAL_PARAMS[mat]
        w_res = p['wp'] / np.sqrt(p['eps_b'] + 1 + 2 * eps_h)
        resonances.append(w_res)
    w_min = min(resonances) * (1 - pad_low)
    w_max = max(resonances) * (1 + pad_high)
    print(f"Auto energy range: {w_min:.2f} – {w_max:.2f} eV")
    for mat, r in zip(config['materials'], resonances):
        print(f"  {mat}: resonance ≈ {r:.2f} eV")
    w = np.linspace(w_min, w_max, n_points) * eV / hbar

    # ── E-field ──
    efield = EField(
        E0=1,
        k_hat=v_normalize(config['k_vec']),
        e_hat=v_normalize(config['e_vec']),
    )

    # ── Dielectric model ──
    drude_model = config['model']
    print(f"Dielectric model: {drude_model}")

    # ── BCM objects ──
    BCM_objects = [
        BCMObject(
            label=f'Sphere{i}_{mat}',
            diameter=D, lmax=lmax,
            eps=_make_eps_drude(mat, D, model=drude_model, material_params=MATERIAL_PARAMS),
            position=np.array(pos),
        )
        for i, (pos, D, mat) in enumerate(
            zip(config['positions'], config['diameters'], config['materials']), start=1
        )
    ]

    Np = len(BCM_objects)

    # ── Interaction matrices ──
    Gi = [None] * Np
    G0 = [[None] * Np for _ in range(Np)]
    Sv = [None] * Np
    for in_idx in trange(Np, desc="Assembling Gi/G0/Sv", unit="obj"):
        Gi[in_idx] = Ginternal(BCM_objects[in_idx])
        for jn_idx in range(Np):
            G0[in_idx][jn_idx] = Gexternal(BCM_objects[in_idx], BCM_objects[jn_idx])
        Sv[in_idx] = Efield_coupling(BCM_objects[in_idx], efield)

    # ── Frequency sweep ──
    dx_max   = lmax * (lmax + 1) + (lmax + 1) - 1
    obj_coef = [np.zeros((dx_max, len(w)), dtype=complex) for _ in range(Np)]
    Sw       = [None] * Np

    pbar = tqdm(range(len(w)), desc="Solving BCM over ω", unit="pt")
    for il in pbar:
        wi = w[il]
        pbar.set_postfix_str(f"E={wi*hbar/eV:.2f} eV")
        try:
            c, Si = solve_BCM(wi, eps_h, BCM_objects, efield, Gi, G0, Sv)
            if all(np.all(np.isfinite(ci)) for ci in c):
                for in_idx in range(Np):
                    obj_coef[in_idx][:, il] = c[in_idx]
                    if il == 0:
                        Sw[in_idx] = np.zeros((len(Si[in_idx]), len(w)), dtype=complex)
                    Sw[in_idx][:, il] = Si[in_idx]
        except Exception:
            pass

    lam_um = 2 * np.pi * 3e14 / w
    for idx_obj in range(Np):
        BCM_objects[idx_obj].set_coefficients(lam_um, obj_coef[idx_obj])

    # ── Power spectra ──
    Psca, Pabs = EM_power(w, eps_h, Gi, G0, BCM_objects)
    Pabs = [np.where(np.isfinite(p), p, 0.0) for p in Pabs]
    Psca = [np.where(np.isfinite(p), p, 0.0) for p in Psca]

    I0    = efield.E0**2 / (2 * Z0_val)
    R_ref = BCM_objects[0].diameter / 2.0
    A_ref = np.pi * R_ref**2

    Qabs_total = np.array(Pabs).sum(axis=0) / (I0 * A_ref)
    Qsca_total = np.array(Psca).sum(axis=0) / (I0 * A_ref)
    Qext_total = Qabs_total + Qsca_total

    # ── Peak detection ──
    _thr = 0.05 * np.nanmax(np.abs(Qabs_total))
    peak_idx_total, _ = find_peaks(Qabs_total, prominence=_thr)
    print(f"Absorption peaks: {len(peak_idx_total)}  at λ = "
          + ", ".join(f"{lam_um[i]*1e3:.1f} nm" for i in peak_idx_total))

    peak_idx_per_particle = []
    for i, obj in enumerate(BCM_objects):
        Qabs_i = Pabs[i] / (I0 * np.pi * (obj.diameter / 2.0)**2)
        thr_i = 0.05 * np.nanmax(np.abs(Qabs_i))
        pidx, _ = find_peaks(Qabs_i, prominence=thr_i)
        peak_idx_per_particle.append(pidx)

    # ── Output folder ──
    outdir = make_results_folder(BCM_objects, efield)
    print("Saving to:", outdir)

    E_eV = w * hbar / eV

    return {
        'BCM_objects': BCM_objects,
        'efield': efield,
        'w': w,
        'lam_um': lam_um,
        'E_eV': E_eV,
        'eps_h': eps_h,
        'I0': I0,
        'Z0': Z0_val,
        'Psca': Psca,
        'Pabs': Pabs,
        'Qsca_total': Qsca_total,
        'Qabs_total': Qabs_total,
        'Qext_total': Qext_total,
        'peak_idx_total': peak_idx_total,
        'peak_idx_per_particle': peak_idx_per_particle,
        'outdir': outdir,
    }