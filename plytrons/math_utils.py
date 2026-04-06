"""
math_utils.py

Utility functions for spherical harmonics, Bessel functions, Legendre polynomials,
and related mathematical operations, optimized for semi-analytical modeling of 
plasmonic systems.

- Associated Legendre polynomials use the SHTOOLS normalization, which differs from
  scipy.special.lpmv by a known scaling factor and (optionally) the Condon–Shortley phase:

      scale = (-1)**m * sqrt((2l + 1) * scipy.special.poch(l + m + 1, -2*m))
      ours = Plm(l, m, z)
      ref = scale * scipy.special.lpmv(m, l, z)

- Spherical harmonics follow quantum mechanics conventions, but check for
  normalization and phase if comparing to SciPy or other references.

References:
-----------
- Holmes and Featherstone, J. Geodesy, 76, 279–299, 2002
- SHTOOLS: https://github.com/SHTOOLS/SHTOOLS
- scipy.special documentation

"""

import numpy as np
from scipy.special import jv, kv
from scipy.signal import find_peaks
import numba as nb
from numba import int32
from numba.types import float64
from numba.experimental import jitclass
import logging
import datetime as dt

hbar = 0.6582118      # Reduced planck constant (eV*fs)
me   = 5.686          # electron mass (eV*fs^2/nm^2)
eps0 = 0.055263493756 # vacuum permittivity (e/V nm)
e    = 1              # electron charge (eV) --- NOT SURE OF THIS

@nb.njit(parallel=False)
def nb_meshgrid(x, y):
    """
    Numba-compatible meshgrid for 1D arrays.

    Parameters
    ----------
    x : ndarray
        1D array representing x-coordinates of the grid.
    y : ndarray
        1D array representing y-coordinates of the grid.

    Returns
    -------
    xx : ndarray
        2D array where each row is a copy of x.
    yy : ndarray
        2D array where each column is a copy of y.
    """
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for j in nb.prange(y.size):
        for k in nb.prange(x.size):
            xx[j,k] = x[k]  # change to x[k] if indexing xy
            yy[j,k] = y[j]  # change to y[j] if indexing xy
    return xx, yy
    
# logging.info(f'{dt.datetime.now()} nb_meshgrid function defined')

#----------------------------------| Spherical Bessel First Kind and real argument |-------------------
@nb.vectorize('float64(int64, float64)', target = "cpu", nopython = True)
def _vec_js_real(l, x):
    '''
    Spherical Bessel function of the first kind of real order and real argument.

    Parameters
    ----------
    l : array_like
        Order (int).
    x : array_like
        Argument (float or complex). 
    '''
    return np.sqrt(np.pi / (2 * x + 1E-10)) * jv(l + 1/2, x)

@nb.njit(['float64(int64, float64)','float64[:](int64, float64[:])','float64[:,:](int64, float64[:,:])'])
def js_real(l: int, x: float) -> float:
    """
    Vectorized spherical Bessel function of the first kind (j_l) for real arguments.

    Computes the spherical Bessel function of the first kind for given order `l` and
    real argument `x`, using the relationship with the cylindrical Bessel function.

    Parameters
    ----------
    l : int or array_like of int
        Order of the spherical Bessel function.
    x : float or array_like of float
        Argument of the spherical Bessel function.

    Returns
    -------
    out : float or ndarray
        Value(s) of the spherical Bessel function j_l(x).
    """
    return _vec_js_real(l, x)

#----------------------------------| Spherical Bessel third Kind and imaginary argument |-------------------
@nb.vectorize('complex128(int64, float64)', target = 'cpu', nopython=True)
def _vec_hs_imag(l, x):
    '''
    Spherical Bessel function of the third kind for imaginary arugments.

    Parameters
    ----------
    l : array_like
        Order (int).
    x : array_like
        Imaginary part of "x" (float). 
    '''
    l = l+1/2
    hv_cplx = 2/np.pi*np.exp(-((l+1)%4)*np.pi*1j/2)*kv(l,x)
    return np.sqrt(np.pi / (2*x*1j)) *hv_cplx

@nb.njit(['complex128(int64, float64)','complex128[:](int64, float64[:])','complex128[:,:](int64, float64[:,:])'])
def hs_imag(l:int, x:float) -> float:
    """
    Spherical Hankel (Bessel) function of the third kind for imaginary arguments.

    Computes the spherical Bessel function of the third kind (Hankel function)
    for order `l` and purely imaginary argument `x`. Returns the function evaluated
    using the relationship with the modified Bessel function of the second kind.

    Parameters
    ----------
    l : int or array_like of int
        Order of the spherical Bessel function.
    x : float or array_like of float
        Imaginary part of the argument (i.e., for an input of the form x * 1j).

    Returns
    -------
    out : complex or ndarray of complex
        Value(s) of the spherical Hankel (Bessel) function for imaginary arguments.
    """
    return _vec_hs_imag(l, x)

#------------------------------------------| Legendre Polynomials |------------------------------------
spec = [
    ('lmax', nb.int64),
    ('sqr', nb.float64[:]),
    ('f1', nb.float64[:]),
    ('f2', nb.float64[:])
]

@jitclass(spec)
class Legendre_poly:
    def __init__(self):
        self.lmax = 0

    def _compute_prefactors(self):
        '''
        Compute multiplicative prefactors used in recursion relationships
           Plmbar(l,m) = x*f1(l,m)*Plmbar(l-1,m) - Plmbar(l-2,m)*f2(l,m)
           k = l*(l+1)/2 + m + 1
        Note that prefactors are not used for the case when m=l and m=l-1,
        as a different recursion is used for these two values. 
        '''
        dp = np.float64
        lmax = self.lmax
        
        f1 = np.empty((lmax+1)*(lmax+2)//2, dtype=dp)
        f2 = np.empty((lmax+1)*(lmax+2)//2, dtype=dp)

        # Precompute square roots of integers that are used several times.
        sqr = np.sqrt(np.arange(1, 2*lmax+2, dtype=dp))
        
        # Compute multiplicative factors
        k = 3
        for l in range(2, lmax+1):
            k += 1
            f1[k-1] = sqr[2*l-2] * sqr[2*l] / dp(l)
            f2[k-1] = dp(l-1) * sqr[2*l] / sqr[2*l-4] / dp(l)
    
            for m in range(1, l-1):
                k += 1
                f1[k-1] = sqr[2*l] * sqr[2*l-2] / sqr[l+m-1] / sqr[l-m-1]
                f2[k-1] = sqr[2*l] * sqr[l-m-2] * sqr[l+m-2] / \
                            sqr[2*l-4] / sqr[l+m-1] / sqr[l-m-1]
    
            k += 2
            
        self.f1, self.f2, self.sqr = f1, f2, sqr

    def Plm(self, lmax, mi, z, csphase=-1, cnorm=1):
        """
        Normalized associated Legendre functions up to degree lmax. The functions are 
        initially scaled by 10^280 sin^m in order to minimize the effects of underflow 
        at large m near the poles (1). The scaled portion of the algorithm will not overflow
        for degrees less than or equal to 2800.
    
        This code is a literal translation of PlmBar.f95 (2), part of the SHTools 
        project (3). 
        
        References:
         1. Holmes and Featherstone 2002, J. Geodesy, 76, 279-299
         2. https://github.com/SHTOOLS/SHTOOLS/blob/develop/src/PlmBar.f95
         3. Wieczorek, M. A., & Meschede, M. (2018). SHTools: Tools for working with
           spherical harmonics. Geochemistry, Geophysics, Geosystems,
        
        Parameters:
            lmax:       int
                        Maximum spherical harmonic degree to compute.
            
            z:          float, ndarray     
                        Polynomial argument.
                    
            csphase     int (optional)
                        Condon-Shortley phase factor of (-1)^m:
                             1: Do not include phase factor (default).
                            -1: Apply phase factor.
            
            cnorm       int (optional)
                        Defines the normalization value of int(-1,1) Plm
                            1: Use complex normalization (default). int(-1,1) Plm = 2
                            0: Use real normalization. int(-1,1) Plm = (2 - delta(0,m))
        
        Returns
            p:          float, ndarray           
                        A vector of all associated Legendre polynomials,
                        evaluated at z up to lmax.
        """
        # some value check before running the code
        assert abs(csphase)==1, "csphase must be 1 (exclude) or -1 (include)."
        assert lmax >= 0       , f'lmax must be greater than or equal to 0.\nInput value is {lmax}'
        assert abs(z) <= 1.0  , f"abs(z) must be less than or equal to 1."
        assert mi >= 0        ,'index m must be >= 0'
    
        # Constants
        phase = 1
        if csphase == -1:
            phase = -1
        
        scalef = 1.0e-280

        if lmax > self.lmax:
            self.lmax = lmax
            self._compute_prefactors()

        f1, f2, sqr = self.f1, self.f2, self.sqr
        
        # Allocate memory
        p = np.empty(((lmax+1)*(lmax+2)//2), dtype=np.float64)
        
        # Calculate P(l,0). These are not scaled.
        u = np.sqrt((1.0 - z) * (1.0 + z))  # sin(theta)
        p[0] = 1.0
    
        if lmax > 0:
            p[1] = sqr[2] * z
            
            k = 2
            for l in range(2, lmax+1):
                k += l
                p[k-1] = f1[k-1] * z * p[k-l-1] - f2[k-1] * p[k-2*l]
        
            # Calculate P(m,m), P(m+1,m), and P(l,m)
            if cnorm == 1:
                pmm = scalef
            else:
                pmm = sqr[1] * scalef
                
            rescalem = 1.0 / scalef
            kstart = 1
            
            for m in range(1, lmax):
                rescalem *= u
        
                # Calculate P(m,m)
                kstart += m + 1
                pmm = phase * pmm * sqr[2*m] / sqr[2*m-1]
                p[kstart-1] = pmm
        
                # Calculate P(m+1,m)
                k = kstart + m + 1
                p[k-1] = z * sqr[2*m+2] * pmm
        
                # Calculate P(l,m)
                for l in range(m+2, lmax+1):
                    k += l
                    p[k-1] = z * f1[k-1] * p[k-l-1] - f2[k-1] * p[k-2*l]
                    p[k-2*l] *= rescalem
        
                p[k-1] *= rescalem
                p[k-lmax-1] *= rescalem
        
            # Calculate P(lmax,lmax)
            rescalem *= u
            
            kstart += m + 1
            p[kstart] = phase * pmm * sqr[2*lmax] / sqr[2*lmax-1] * rescalem
    
        return p[(lmax*(lmax+1))//2+mi]

@nb.vectorize(['float64(int64, int64, float64)'], cache=False)
def nb_lpmv(l, m, z):
    '''
    Associated Legendre polynomials with Condon–Shortley phase factor (-1)**m.
    The polynomials are normalized by:
         sqrt((2l + 1)(l - m)!/(l + m)!)
    
    Parameters
    ------------
        - l: int
             degree of the polynomial
        - m: int
             order of the polynomial
        - z: float(ndarray)
             argument
    Returns
    ------------
        Evaluated polynomial Plm(z)
    '''
    poly = Legendre_poly() # set a Legendre Polynomial class 
    return poly.Plm(l, m, z, csphase = -1)

@nb.vectorize('complex128(float64, float64, float64, float64)', target='cpu', nopython=True, cache = False)
def qm_sph_harm(m, l, theta, phi):
    """
    Quantum spherical harmonics. Condon–Shortley phase factor (-1)**m is implicit in 
    Associated Legendre polynomials "nb_lpmv".
    
    * Note: this function is equivalent to scipy.special.sph_harm
                sph_harm(m, l, phi, theta)
    
    Parameters
    ----------
    m     : int
            Order of the harmonic (int); must have |m| <= l.
    l     : int
            Degree of the harmonic (int); must have l >= 0. 
    phi   : array_like
            Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    theta : array_like
            Polar (colatitudinal) coordinate; must be in [0, pi].
        
    Returns
    -------
    y_ml  : complex array
            complex scalar or ndarray
    """
    
    mp = abs(m)
    assert mp <= l , 'm cannot not be greater than l'
    assert l >= 0,   'l cannot not be negative'
    
    x = np.cos(theta)

    #  Associated legendre polynomial (with Condon-Shortley phase factor)
    y_ml = nb_lpmv(nb.int64(l), nb.int64(mp), x)
    
    y_ml *= np.sqrt(1/ (4.0 * np.pi)) # 4pi normalization
    y_ml *= np.exp(1j * mp * phi)      # azimuthal component

    if m < 0:
        # y_ml = - y_ml
        y_ml = (-1)**mp*np.conj(y_ml)
    return y_ml

@nb.vectorize('complex128(float64, float64, float64, float64)', target='cpu', nopython=True, cache = False)
def em_sph_harm(m, l, theta, phi):
    """
    Electromagnetic spherical harmonics without Condon–Shortley phase factor (-1)**m
    
    * Note: this function is equivalent to scipy.special.sph_harm
                sph_harm(m, l, phi, theta)
    
    Parameters
    ----------
    m     : int
            Order of the harmonic (int); must have |m| <= l.
    l     : int
            Degree of the harmonic (int); must have l >= 0. 
    phi   : array_like
            Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    theta : array_like
            Polar (colatitudinal) coordinate; must be in [0, pi].
        
    Returns
    -------
    y_ml  : complex array
            complex scalar or ndarray
    """
    
    mp = abs(m)
    assert mp <= l , 'm cannot not be greater than l'
    assert l >= 0,   'l cannot not be negative'
    
    return (-1)**(-m)*qm_sph_harm(m, l, theta, phi)

#------------------------------------------| Wigner 3-j Symbols |--------------------------------------

# referencia
@nb.jit("float64(int32, int32, int32, int32)")
def A(j, j2, j3, m1):
    return np.sqrt((j**2 - (j2 - j3)**2) * ((j2 + j3 + 1)**2 - j**2) * (j**2 - m1**2))

@nb.jit("int32(int32, int32, int32, int32, int32)")
def Yf(j, j2, j3, m2, m3):
    return (2 * j + 1) * ((m2 + m3) * (j2 * (j2 + 1) - j3 * (j3 + 1)) - (m2 - m3) * j * (j + 1))

@nb.jit("float64(int32, int32, int32, int32)")
def Xf(j, j2, j3, m1):
    return j * A(j+1, j2, j3, m1)

@nb.jit("float64(int32, int32, int32, int32)")
def Zf(j, j2, j3, m1):
    return (j+1) * A(j, j2, j3, m1)

@nb.jit("void(float64[:], int32, int32)")
def normalize(f, j_min, j_max):
    norm = 0.0
    for j in range(j_min, j_max+1):
        norm += (2 * j + 1) * f[j] ** 2
    f[j_min:j_max + 1] /= np.sqrt(norm)

@nb.jit("void(float64[:], int32, int32, int32, int32, int32, int32)")
def determine_signs(f, j_min, j_max, j2, j3, m2, m3):
    if (f[j_max] < 0.0 and (-1)**(j2-j3+m2+m3) > 0) or (f[j_max] > 0.0 and (-1)**(j2-j3+m2+m3) < 0):
        f[j_min:j_max+1] *= -1.0

@jitclass([
    ('_size', int32),
    ('workspace', float64[:]),
])
class Wigner3jCalculator(object):
    def __init__(self, j2_max, j3_max):
        self._size = j2_max + j3_max + 1
        self.workspace = np.empty(4 * self._size, dtype=np.float64)

    @property
    def size(self):
        return self._size

    def calculate(self, j2, j3, m2, m3):
        """Compute Wigner 3-j symbols

        For given values of j₂, j₃, m₂, m₃, this computes all valid values of

            ⎛j₁  j₂  j₃⎞   _   ⎛j₂  j₃  j₁⎞   _   ⎛j₃  j₁  j₂⎞
            ⎝m₁  m₂  m₃⎠   -   ⎝m₂  m₃  m₁⎠   -   ⎝m₃  m₁  m₂⎠

        The valid values have m₁=-m₂-m₃ and j₁ ranging from max(|j₂-j₃|, |m₁|) to j₂+j₃.
        The calculation uses the approach described by Luscombe and Luban (1998)
        <https://doi.org/10.1103/PhysRevE.57.7274>, which is a recurrence method, leading
        to significant gains in speed and accuracy.

        The returned array is a slice of this object's `workspace` array, and so will not
        remain the same between calls to this function.  If you want to keep a copy of
        the results, explicitly call the `numpy.copy` method.

        The returned array is indexed by j₁.  In particular, note that some invalid
        j₁ indices are accessible, but have value 0.0.

        This implementation uses several tricks gleaned from the Fortran code in
        <https://github.com/SHTOOLS/SHTOOLS>, which also implements the Luscombe-Luban
        algorithm.  In particular, that code (and now this code) treats several special
        cases that were not clearly specified by Luscombe-Luban.

        To use this object, do something like this:

            # Try to do this just once because it allocates memory, which is slow
            calculator = Wigner3jCalculator(j2_max, j3_max)

            # Presumably, the following is inside some loop over j2, j3, m2, m3
            w3j = calculator.calculate(j2, j3, m2, m3)
            m1 = - m2 - m3
            for j1 in range(max(abs(j2-j3), abs(m1)), j2+j3+1):
                w3j[j1]  # This is the value of the 3-j symbol written above

        Again, the array w3j contains accessible memory outside of the bounds of j1 given
        in the above loop, but those values will all be 0.0.

        """
        m1 = -(m2 + m3)

        undefined_min = False
        undefined_max = False
        scale_factor = 1_000.0

        # Set up the workspace
        self.workspace[:] = 0.0
        f = self.workspace[:self.size]
        sf = self.workspace[self.size:2*self.size]
        rf = sf  # An alias for the same memory as `sf`
        F_minus = self.workspace[2*self.size:3*self.size]
        F_plus = self.workspace[3*self.size:4*self.size]

        # Calculate some useful bounds
        j_min = max(abs(j2-j3), abs(m2+m3))
        j_max = j2 + j3

        # Skip all calculations if there are no nonzero elements
        if abs(m2) > j2 or abs(m3) > j3:
            return f
        if j_max < j_min:
            return f

        # When only one term is present, we have a simple formula
        if j_max == j_min:
            f[j_min] = 1.0 / np.sqrt(2.0 * j_min + 1.0)
            if (f[j_min] < 0.0 and (-1) ** (j2-j3+m2+m3) > 0) or (f[j_min] > 0.0 and (-1) ** (j2-j3+m2+m3) < 0):
                f[j_min] *= -1.0
            return f

        # Forward iteration over first "nonclassical" region j_min ≤ j ≤ j_minus

        Xf_j_min = Xf(j_min, j2, j3, m1)
        Yf_j_min = Yf(j_min, j2, j3, m2, m3)

        if m1 == 0 and m2 == 0 and m3 == 0:
            # Recurrence is undefined, but it's okay because all odd terms must be zero
            F_minus[j_min] = 1.0
            F_minus[j_min+1] = 0.0
            j_minus = j_min + 1

        elif Yf_j_min == 0.0:
            # The second term is either undefined or zero
            if Xf_j_min == 0.0:
                undefined_min = True
                j_minus = j_min
            else:
                F_minus[j_min] = 1.0
                F_minus[j_min+1] = 0.0
                j_minus = j_min + 1

        elif Xf_j_min * Yf_j_min >= 0.0:
            # The second term is already in the classical region
            F_minus[j_min] = 1.0
            F_minus[j_min+1] = -Yf_j_min / Xf_j_min
            j_minus = j_min + 1

        else:
            # Use recurrence relation, Eq. (3) from Luscombe and Luban (1998)
            sf[j_min] = -Xf_j_min / Yf_j_min
            j_minus = j_max
            for j in range(j_min+1, j_max):
                denominator = Yf(j, j2, j3, m2, m3) + Zf(j, j2, j3, m1) * sf[j - 1]
                Xf_j = Xf(j, j2, j3, m1)
                if abs(Xf_j) > abs(denominator) or Xf_j * denominator >= 0.0 or denominator == 0.0:
                    j_minus = j - 1
                    break
                else:
                    sf[j] = -Xf_j / denominator
            F_minus[j_minus] = 1.0
            for k in range(1, j_minus - j_min + 1):
                F_minus[j_minus - k] = F_minus[j_minus - k + 1] * sf[j_minus - k]
            if j_minus == j_min:
                # Calculate at least two terms so that these can be used in three-term recursion
                F_minus[j_min+1] = -Yf_j_min / Xf_j_min
                j_minus = j_min + 1

        if j_minus == j_max:
            # We're finished!
            f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]
            normalize(f, j_min, j_max)
            determine_signs(f, j_min, j_max, j2, j3, m2, m3)
            return f

        # Reverse iteration over second "nonclassical" region j_plus ≤ j ≤ j_max

        Yf_j_max = Yf(j_max, j2, j3, m2, m3)
        Zf_j_max = Zf(j_max, j2, j3, m1)

        if m1 == 0 and m2 == 0 and m3 == 0:
            # Recurrence is undefined, but it's okay because all odd terms must be zero
            F_plus[j_max] = 1.0
            F_plus[j_max-1] = 0.0
            j_plus = j_max - 1

        elif Yf_j_max == 0.0:
            # The second term is either undefined or zero
            if Zf_j_max == 0.0:
                undefined_max = True
                j_plus = j_max
            else:
                F_plus[j_max] = 1.0
                F_plus[j_max-1] = - Yf_j_max / Zf_j_max
                j_plus = j_max - 1

        elif Yf_j_max * Zf_j_max >= 0.0:
            # The second term is already in the classical region
            F_plus[j_max] = 1.0
            F_plus[j_max-1] = - Yf_j_max / Zf_j_max
            j_plus = j_max - 1

        else:
            # Use recurrence relation, Eq. (2) from Luscombe and Luban (1998)
            rf[j_max] = -Zf_j_max / Yf_j_max
            j_plus = j_min
            for j in range(j_max-1, j_minus - 1, -1):
                denominator = Yf(j, j2, j3, m2, m3) + Xf(j, j2, j3, m1) * rf[j + 1]
                Zf_j = Zf(j, j2, j3, m1)
                if denominator == 0.0 or abs(Zf_j) > abs(denominator) or Zf_j * denominator >= 0.0:
                    j_plus = j + 1
                    break
                else:
                    rf[j] = -Zf_j / denominator
            F_plus[j_plus] = 1.0
            for k in range(1, j_max - j_plus + 1):
                F_plus[j_plus + k] = F_plus[j_plus + k - 1] * rf[j_plus + k]
            if j_plus == j_max:
                F_plus[j_max-1] = - Yf_j_max / Zf_j_max
                j_plus = j_max - 1

        # Three-term recurrence over "classical" region j_minus ≤ j ≤ j_plus

        if undefined_min and undefined_max:
            raise ValueError("Cannot initialize recurrence in Wigner3jCalculator.calculate")

        if not undefined_min and not undefined_max:
            # Iterate upwards and downwards, meeting in the middle
            j_mid = (j_minus + j_plus) // 2
            for j in range(j_minus, j_mid):
                F_minus[j+1] = (
                    - (Yf(j, j2, j3, m2, m3) * F_minus[j] + Zf(j, j2, j3, m1) * F_minus[j-1]) / Xf(j, j2, j3, m1)
                )
                if abs(F_minus[j+1]) > 1.0:
                    F_minus[j_min:j+1+1] /= scale_factor
                if abs(F_minus[j+1] / F_minus[j-1]) < 1.0 and F_minus[j+1] != 0.0:
                    j_mid = j + 1
                    break
            F_minus_j_mid = F_minus[j_mid]
            if F_minus[j_mid - 1] != 0.0 and abs(F_minus_j_mid / F_minus[j_mid - 1]) < 1.0e-6:
                j_mid -= 1
                F_minus_j_mid = F_minus[j_mid]
            for j in range(j_plus, j_mid, -1):
                F_plus[j-1] = (
                    - (Xf(j, j2, j3, m1) * F_plus[j+1] + Yf(j, j2, j3, m2, m3) * F_plus[j]) / Zf(j, j2, j3, m1)
                )
                if abs(F_plus[j-1]) > 1.0:
                    F_plus[j-1:j_max+1] /= scale_factor
            F_plus_j_mid = F_plus[j_mid]
            if j_mid == j_max:
                f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]
            elif j_mid == j_min:
                f[j_min:j_max + 1] = F_plus[j_min:j_max + 1]
            else:
                f[j_min:j_mid + 1] = F_minus[j_min:j_mid + 1] * F_plus_j_mid / F_minus_j_mid
                f[j_mid + 1:j_max + 1] = F_plus[j_mid + 1:j_max + 1]

        elif not undefined_min and undefined_max:
            # Iterate upwards only
            for j in range(j_minus, j_plus):
                F_minus[j + 1] = (
                    - (Zf(j, j2, j3, m1) * F_minus[j - 1] + Yf(j, j2, j3, m2, m3) * F_minus[j]) / Xf(j, j2, j3, m1)
                )
                if abs(F_minus[j + 1]) > 1:
                    F_minus[j_min:j+1+1] /= scale_factor
            f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]

        elif undefined_min and not undefined_max:
            # Iterate downwards only
            for j in range(j_plus, j_min, -1):
                F_plus[j-1] = (
                    - (Xf(j, j2, j3, m1) * F_plus[j+1] + Yf(j, j2, j3, m2, m3) * F_plus[j]) / Zf(j, j2, j3, m1)
                )
                if abs(F_plus[j-1]) > 1:
                    F_plus[j-1:j_max+1] /= scale_factor
            f[j_min:j_max + 1] = F_plus[j_min:j_max + 1]

        normalize(f, j_min, j_max)
        determine_signs(f, j_min, j_max, j2, j3, m2, m3)
        return f


@nb.jit('f8(i8,i8,i8,i8,i8,i8)')
def Wigner3j(j_1, j_2, j_3, m_1, m_2, m_3):
    """Calculate the Wigner 3-j symbol

    NOTE: If you are calculating more than one value, you probably want to use the
    Wigner3jCalculator object.  This function uses that object inefficiently because, in
    computing one particular value, that object uses recurrence relations to compute
    numerous nearby values that you will probably need to compute anyway.

    The result is what is normally represented as

        ⎛j₁  j₂  j₃⎞
        ⎝m₁  m₂  m₃⎠

    The inputs must be integers.  (Half integer arguments are sacrificed so that we can
    use numba.)  Nonzero return quantities only occur when the `j`s obey the triangle
    inequality (any two must add up to be as big as or bigger than the third).

    Examples
    ========

    >>> from spherical_functions import Wigner3j
    >>> Wigner3j(2, 6, 4, 0, 0, 0)
    0.186989398002
    >>> Wigner3j(2, 6, 4, 0, 0, 1)
    0

    """
    if m_1 + m_2 + m_3 != 0:
        return 0.0
    if abs(m_1) > j_1 or abs(m_2) > j_2 or abs(m_3) > j_3:
        return 0.0
    # Permute cyclically to ensure that j_1 is the largest
    if j_1 == max(j_1, j_2, j_3):
        pass
    elif j_2 == max(j_1, j_2, j_3):
        j_1, j_2, j_3 = j_2, j_3, j_1
        m_1, m_2, m_3 = m_2, m_3, m_1
    else:  # j_3 == max(j_1, j_2, j_3)
        j_1, j_2, j_3 = j_3, j_1, j_2
        m_1, m_2, m_3 = m_3, m_1, m_2
    if j_1 > j_2 + j_3:
        return 0.0
    calculator = Wigner3jCalculator(j_2, j_3)
    w3j = calculator.calculate(j_2, j_3, m_2, m_3)
    return w3j[j_1]


@nb.jit('f8(i8,i8,i8,i8,i8,i8)')
def clebsch_gordan(j_1, m_1, j_2, m_2, j_3, m_3):
    """Calculate the Clebsch-Gordan coefficient <j1 m1 j2 m2 | j3 m3>

    NOTE: If you are calculating more than one value, you probably want to use the
    Wigner3jCalculator object.  This function uses that object inefficiently because, in
    computing one particular value, that object uses recurrence relations to compute
    numerous nearby values that you will probably need to compute anyway.

    """
    return (-1.)**(-j_1+j_2-m_3) * np.sqrt(2*j_3+1) * Wigner3j(j_1, j_2, j_3, m_1, m_2, -m_3)

@nb.jit('f8(i8,i8,i8,i8,i8,i8)')
def gaunt_coeff(l1,l2,l3,m1,m2,m3):
    '''3 spherical harmonics integration using wigner3j symbols'''
    return np.sqrt(float((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi)))*Wigner3j(l1,l2,l3,0,0,0)*Wigner3j(l1,l2,l3,m1,m2,m3)


def detect_peaks(x_data, y_data, prominence=0.1, width=None, height=None,print_data=True):
    """
    Detect peaks in the data and return their x and y values.
    
    Parameters:
    -----------
    x_data : array-like
        The x-coordinates of the data
    y_data : array-like
        The y-coordinates of the data
    prominence : float, optional
        The prominence threshold for peak detection
    width : float, optional
        The width threshold for peak detection
    height : float, optional
        The height threshold for peak detection
    plot_result : bool, optional
        Whether to plot the result with the detected peaks
        
    Returns:
    --------
    peak_x : array
        The x-coordinates of the detected peaks
    peak_y : array
        The y-coordinates of the detected peaks
    """
    # Find peaks using scipy's find_peaks function
    peaks, properties = find_peaks(y_data, prominence=prominence, width=width, height=height)
    
    # Get x and y values of the peaks
    peak_x = x_data[peaks]
    peak_y = y_data[peaks]

    if print_data:    
        # Print the detected peaks
        print("Detected peaks:")
        for i in range(len(peak_x)):
            print(f"Peak {i+1}: x = {peak_x[i]:.3f} eV, y = {peak_y[i]:.3f} (10^-2)/eV/(ps·nm²)")
    
    return peak_x, peak_y
