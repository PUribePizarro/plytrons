# plytrons/plot_utils.py

from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from plytrons.bcm_sphere import get_axis as _bcm_get_axis
import plytrons.bcm_sphere as bcm
from scipy.constants import hbar, eV


__all__ = ["make_results_folder", "axis_label",
           "find_dominant_peak",
           "combined_absorption_hotcarriers_gif", "static_multi_resonance_grid_gif",
           "get_preset_geometry", "build_cluster", "get_polarizations"]


def axis_label(v):
    """Return 'x', 'y', or 'z' if v is axis-aligned, else '(x,y,z)' components."""
    try:
        return _bcm_get_axis(v)
    except Exception:
        v = np.asarray(v, float).ravel()
        return f'({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
import numpy as np
import plytrons.bcm_sphere as bcm  # for get_axis

def path_with_polarization(out_path, efield, *,
                           key="E",                # label used in the tag → 'E'
                           default_basename="figure",
                           default_ext=".png"):
    """
    Append polarization tag from efield.e_hat to a path.
    - If `out_path` is a filename → 'name_Ey.png'
    - If `out_path` is a directory or None/"" → '<default_basename>_Ey.png' inside it

    Returns a pathlib.Path.
    """
    def _axis_from_vec(v):
        try:
            return bcm.get_axis(v)        # expected: 'x', 'y', or 'z'
        except Exception:
            v = np.asarray(v, float).ravel()
            if v.size != 3:
                raise ValueError("efield.e_hat must be a 3-vector.")
            return ("x", "y", "z")[int(np.argmax(np.abs(v)))]

    axis = _axis_from_vec(efield.e_hat)
    tag  = f"_{key}{axis}"

    p = Path("." if not out_path else out_path)
    if p.suffix:  # looks like a file → insert before extension
        return p.with_name(p.stem + tag + p.suffix)
    else:         # looks like a directory → create a filename inside
        return p / f"{default_basename}{tag}{default_ext}"


# ===================== utilidades de geometría =====================
def set_axes_equal(ax):
    """Escalas iguales en x, y, z (esferas redondas)."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    R = max(np.diff(xlim)[0], np.diff(ylim)[0], np.diff(zlim)[0]) / 2
    cx, cy, cz = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)
    ax.set_zlim(cz - R, cz + R)

def sphere_mesh(center, radius, nu=96, nv=48):
    """Malla (X,Y,Z) y normales (Nx,Ny,Nz) para una esfera."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"El radio debe ser positivo y finito; recibido {radius!r}")

    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    U, V = np.meshgrid(u, v, indexing="xy")
    X = cx + radius*np.cos(U)*np.sin(V)
    Y = cy + radius*np.sin(U)*np.sin(V)
    Z = cz + radius*np.cos(V)
    Nx = (X - cx) / radius
    Ny = (Y - cy) / radius
    Nz = (Z - cz) / radius
    return X, Y, Z, Nx, Ny, Nz

def infer_radius_from_centers(C, scale=0.45, fallback=1.0):
    """
    Radio base desde la mínima distancia entre centros.
    Para N=1 o si no hay distancias válidas, usa 'fallback'.
    """
    C = np.asarray(C, dtype=float)
    N = len(C)
    if N <= 1:
        return float(fallback)

    dmin = np.inf
    for i in range(N):
        di = np.linalg.norm(C[i+1:] - C[i], axis=1)
        if len(di):
            dmin = min(dmin, di.min())

    if not np.isfinite(dmin) or dmin <= 0:
        return float(fallback)

    return scale * dmin

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Vector nulo no tiene dirección.")
    return v / n

def _orthogonalize(v, n):
    """Proyecta v al subespacio perpendicular a n."""
    n = _unit(n)
    return v - np.dot(v, n) * n

# ===================== sombreado (Lambert + Phong) =====================
def shaded_facecolors(Nx, Ny, Nz, base_rgb, light_dir=(1, -1, 2),
                      ambient=0.35, diffuse=0.65, specular=0.20, shininess=40):
    """
    Calcula colores RGBA con iluminación difusa + especular sobre un color base.
    base_rgb: (3,) en [0,1]; light_dir se normaliza.
    """
    L = _unit(np.array(light_dir, dtype=float))
    NdotL = np.clip(Nx*L[0] + Ny*L[1] + Nz*L[2], 0.0, 1.0)

    # Reflexión R = 2(N·L)N - L
    Rx = 2*NdotL*Nx - L[0]
    Ry = 2*NdotL*Ny - L[1]
    Rz = 2*NdotL*Nz - L[2]
    Rlen = np.sqrt(Rx*Rx + Ry*Ry + Rz*Rz) + 1e-12
    Rx, Ry, Rz = Rx/Rlen, Ry/Rlen, Rz/Rlen

    # Dirección de vista (aprox cámara)
    V = _unit(np.array([0.3, 0.2, 1.0]))
    RdotV = np.clip(Rx*V[0] + Ry*V[1] + Rz*V[2], 0.0, 1.0)
    spec = (RdotV ** shininess)

    shade = ambient + diffuse * NdotL + specular * spec
    shade = np.clip(shade, 0, 1)

    fc = np.empty(Nx.shape + (4,), dtype=float)
    for c in range(3):
        fc[..., c] = base_rgb[c] * shade
    fc[..., 3] = 1.0
    return fc

# ===================== estilo de ejes 3D (compatible) =====================
def _style_3d_axes(ax,
                   pane_face=(0.97, 0.97, 0.99, 1.0),
                   pane_edge=(0.75, 0.75, 0.85, 1.0),
                   grid_alpha=0.20):
    """Estiliza ejes 3D compatible con distintas versiones de Matplotlib."""
    ax.grid(True, alpha=grid_alpha)
    # API nueva (>= ~3.6)
    try:
        ax.xaxis.set_pane_color(pane_face)
        ax.yaxis.set_pane_color(pane_face)
        ax.zaxis.set_pane_color(pane_face)
        ax.xaxis.line.set_color(pane_edge)
        ax.yaxis.line.set_color(pane_edge)
        ax.zaxis.line.set_color(pane_edge)
        return
    except Exception:
        pass
    # Alternativa con .pane
    try:
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor(pane_face)
            a.pane.set_edgecolor(pane_edge)
        return
    except Exception:
        pass
    # Fallback API antigua (Axes3D legacy)
    try:
        for pane in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
            pane.set_pane_color(pane_face)
            pane.line.set_color(pane_edge)
    except Exception:
        pass

# ===================== flechas EM (con leyenda) =====================
def draw_em_arrows(ax, origin, k_vec, E_vec, lengths,
                   colors=("C3","C0","C2"),
                   labels=(r"$\mathbf{k}$", r"$\mathbf{E}$", r"$\mathbf{H}$"),
                   show_labels_on_plot=True):
    """
    Dibuja flechas k, E, (opcional) H y devuelve 'handles' de leyenda con sus colores.
    """
    k_hat = _unit(k_vec)
    E_hat = _unit(_orthogonalize(E_vec, k_hat))
    H_hat = np.cross(k_hat, E_hat)
    k_len, E_len, H_len = lengths

    # Flecha k
    ax.quiver(*origin, *(k_hat*k_len), arrow_length_ratio=0.12, linewidth=2.2, color=colors[0])
    if show_labels_on_plot:
        ax.text(*(origin + k_hat*(k_len*1.20)), labels[0], color=colors[0])

    # Flecha E
    ax.quiver(*origin, *(E_hat*E_len), arrow_length_ratio=0.12, linewidth=2.2, color=colors[1])
    if show_labels_on_plot:
        ax.text(*(origin + E_hat*(E_len*1.20)), labels[1], color=colors[1])

    # Flecha H (si hay longitud)
    if H_len and H_len > 0:
        ax.quiver(*origin, *(H_hat*H_len), arrow_length_ratio=0.12, linewidth=2.2, color=colors[2])
        if show_labels_on_plot:
            ax.text(*(origin + H_hat*(H_len*1.20)), labels[2], color=colors[2])

    # Proxies para leyenda
    handles = []
    if k_len and k_len > 0:
        handles.append(Line2D([0],[0], color=colors[0], lw=3, label=labels[0]))
    if E_len and E_len > 0:
        handles.append(Line2D([0],[0], color=colors[1], lw=3, label=labels[1]))
    if H_len and H_len > 0:
        handles.append(Line2D([0],[0], color=colors[2], lw=3, label=labels[2]))
    return handles

# ===================== función principal =====================
def plot_spheres(centers, radii=None, *,
                 cmap_name="viridis",
                 light_dir=(1, -1, 2),
                 elev=24, azim=42,
                 title=None,
                 # ----- EM -----
                 k_vec=None, E_vec=None, draw_H=True,
                 origin_mode="centroid", corner=("max","max","max"),
                 corner_inset=0.12, offset_along_k=-0.15,
                 # ----- NUEVO: guardado -----
                 outdir: Optional[Union[str, Path]] = None,
                 fname: str = "geometry.png",
                 save_dpi: int = 300,
                 save_transparent: bool = False,
                 show: bool = True):
    """
    Si 'outdir' no es None, guarda la figura como outdir / fname (PNG por defecto).
    Devuelve (fig, ax).
    """
    """
    origin_mode:
        - "centroid": ancla en el centroide (comportamiento anterior).
        - "corner"  : ancla en una esquina del bounding box (controlada por 'corner').
        - (x,y,z)   : vector con el punto exacto.
    corner:
        Tupla con 'min'/'max' para (x,y,z). Ej: ('min','max','max').
    corner_inset:
        Cuánto desplazar desde la esquina hacia adentro (0–0.5 aprox), como fracción del tamaño de la escena.
    """
    C = np.asarray(centers, dtype=float)
    assert C.ndim == 2 and C.shape[1] == 3, "centers debe ser (N,3)."
    N = len(C)

    # --- radios (igual que antes) ---
    if radii is None:
        r = infer_radius_from_centers(C, scale=0.45, fallback=1.0)
        radii = np.full(N, r, dtype=float)
    elif np.isscalar(radii):
        radii = np.full(N, float(radii), dtype=float)
    else:
        radii = np.asarray(radii, dtype=float)
        assert len(radii) == N, "radii debe ser escalar o de largo N."

    # --- figura y estilo (igual que antes) ---
    fig = plt.figure(figsize=(8.8, 8.6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    _style_3d_axes(ax, pane_face=(0.97, 0.97, 0.99, 1.0),
                      pane_edge=(0.75, 0.75, 0.85, 1.0),
                      grid_alpha=0.20)

    cmap = plt.get_cmap(cmap_name)
    raw = cmap(np.linspace(0.08, 0.92, N))[:, :3]
    base_colors = 0.25 + 0.75 * raw

    for i, (c, r) in enumerate(zip(C, radii)):
        X, Y, Z, Nx, Ny, Nz = sphere_mesh(c, r)
        facecolors = shaded_facecolors(Nx, Ny, Nz, base_colors[i], light_dir=light_dir)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        facecolors=facecolors, linewidth=0.1,
                        antialiased=True, shade=False)
        ax.scatter(*c, s=18, c="k", depthshade=False)

    # --- encuadre robusto (igual que te dejé) ---
    ptp_xyz = np.ptp(C, axis=0, keepdims=True).max()
    r_max = float(np.max(radii))
    scene_span = max(ptp_xyz, 3.0 * r_max, 1e-9)
    pad = 0.30 * scene_span

    cxyz = C.mean(axis=0)
    half = 0.5 * scene_span + pad
    mins = cxyz - half
    maxs = cxyz + half
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title: ax.set_title(title, pad=10)
    fig.patch.set_facecolor("white")
    ax.set_facecolor((0.98, 0.98, 1.0))

    # --- flechas EM + leyenda ---
    legend_handles = []
    if k_vec is not None and E_vec is not None:
        # origen segun modo
        if isinstance(origin_mode, str):
            mode = origin_mode.lower()
            if mode == "centroid":
                o = cxyz.copy()
            elif mode == "corner":
                # corner=("min"/"max", "min"/"max", "min"/"max")
                selects = []
                for i, mm in enumerate(corner):
                    if str(mm).lower().startswith("min"):
                        selects.append(mins[i])
                    elif str(mm).lower().startswith("max"):
                        selects.append(maxs[i])
                    else:
                        raise ValueError("corner debe contener 'min' o 'max' por eje.")
                o = np.array(selects, dtype=float)
                # mete el origen hacia adentro del volumen un poco
                inset = float(corner_inset) * scene_span
                # el vector hacia el centro, normalizado por eje
                to_center = cxyz - o
                # evita mover en 0 si coincide exacto con el centroide
                if np.allclose(to_center, 0):
                    to_center = np.array([1.0, 1.0, 1.0])
                to_center = to_center / (np.linalg.norm(to_center) + 1e-12)
                o = o + inset * to_center
            else:
                raise ValueError("origin_mode debe ser 'centroid', 'corner' o un vector (3,).")
        else:
            o = np.asarray(origin_mode, dtype=float)  # es un vector (x,y,z)
            if o.shape != (3,):
                raise ValueError("origin_mode vector debe tener forma (3,)")

        # mismo offset a lo largo de k
        k_hat = _unit(k_vec)
        o = o + offset_along_k * k_hat

        lengths = (0.36*scene_span, 0.28*scene_span, 0.25*scene_span if draw_H else 0.0)
        legend_handles = draw_em_arrows(
            ax, o, k_vec, E_vec, lengths,
            colors=("C3","C0","C2"),
            labels=(r"$\mathbf{k}$", r"$\mathbf{E}$", r"$\mathbf{H}$"),
            show_labels_on_plot=True
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left",
                  frameon=True, framealpha=0.9)

    # ======== GUARDAR FIGURA (nuevo) ========
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / fname, dpi=save_dpi, bbox_inches="tight",
                    transparent=save_transparent)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax 

def make_results_folder(
    bcms: Sequence,                 # sequence of BCMObject (must have .diameter, .position)
    efield,                         # EField (must have .e_hat and .k_hat)
    *,
    lmax: Optional[int] = None,
    eps_h: Optional[float] = None,
    prefix: Union[str, Path, None] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    include_timestamp: bool = False,
    extra_tag: Optional[str] = None,
) -> Path:
    """
    Create (and return) a results folder whose name encodes the geometry.

    Naming rules:
      - Symmetric dimer/trimer with equal diameters → "{dimer|trimer}_D{:.1f}nm_gap{:.1f}nm_{Exkz}"
      - Otherwise → "{<N>-mer}-{shape}_{Dmix|D{:.1f}nm}_gap{:.1f}nm_{Exkz}"
        where shape ∈ {single, line, tri-equil, tri-iso, tri-scal, planar, ring, 3D}

    Parameters
    ----------
    bcms : sequence of BCMObject
        Each object must expose `.diameter: float` (nm) and `.position: array-like (nm)`.
    efield : EField
        Must expose `.e_hat` and `.k_hat` for orientation labeling via Exkz, Eykx, etc.
    lmax : int, optional
        If provided, appended as `_lmax{lmax}`.
    eps_h : float, optional
        If provided, appended as `_epsh{eps_h}` (compact).
    prefix : str | Path
        Parent directory for the results folder. Created if missing.
    rtol, atol : float
        Tolerances for geometric comparisons (equal diameters, equal sides, etc.).
    include_timestamp : bool
        If True, append a UTC timestamp suffix `_YYYYmmdd-HHMMSS`.
    extra_tag : str, optional
        Any extra descriptor to append at the end (e.g., "run42").

    Returns
    -------
    Path
        The created directory path.

    Examples
    --------
    >>> outdir = make_results_folder(BCM_objects, efield)
    >>> fig.savefig(outdir / "absorption_spectrum.png", dpi=200)
    """
    bcms = list(bcms)
    Np = len(bcms)

    # ----- base name by particle count -----
    nname_map = {1: "monomer", 2: "dimer", 3: "trimer", 4: "tetramer",
                 5: "pentamer", 6: "hexamer", 7: "heptamer"}
    nname = nname_map.get(Np, f"{Np}-mer")

    # ----- auto-detect Results folder when prefix is None ----------------
    if prefix is None:
        _folder_map = {
            1: "Monomer", 2: "Dimer", 3: "Trimer", 4: "Tetramer",
            5: "Pentamer", 6: "Hexamer", 7: "Heptamer", 8: "Octahedron",
        }
        _cluster = _folder_map.get(Np, "N-mer")
        _project_root = Path(__file__).parent.parent
        prefix = _project_root / "Results" / "spherical_NP" / _cluster / "static_simulations"

    # ----- diameters -----
    diam = np.array([float(o.diameter) for o in bcms], dtype=float)
    equal_D = np.allclose(diam, diam[0], rtol=rtol, atol=atol)
    Dpart = f"D{diam[0]:.1f}nm" if equal_D else "Dmix"

    # ----- minimum gap (nm) -----
    if Np >= 2:
        centers = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        gap_min = np.inf
        for i, j in combinations(range(Np), 2):
            dc = float(np.linalg.norm(centers[i] - centers[j]))
            gap = dc - 0.5 * (diam[i] + diam[j])
            if gap < gap_min:
                gap_min = gap
        gap_min = max(0.0, float(gap_min))
        Gpart = f"gap{gap_min:.1f}nm"
    else:
        Gpart = "isolated"

    # ----- simple shape descriptor -----
    if Np == 1:
        shape = "single"
    elif Np == 2:
        # two points are always collinear
        shape = "line"
    elif Np == 3:
        coords = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        centered = coords - coords.mean(axis=0, keepdims=True)
        rank = np.linalg.matrix_rank(centered, tol=1e-9)
        if rank <= 1:
            shape = "line"
        else:
            # classify triangle
            sides = np.array([np.linalg.norm(coords[i] - coords[j])
                              for i, j in combinations(range(3), 2)])
            if np.allclose(sides, sides[0], rtol=rtol, atol=atol):
                shape = "tri-equil"
            elif (np.isclose(sides[0], sides[1], rtol=rtol, atol=atol) or
                  np.isclose(sides[0], sides[2], rtol=rtol, atol=atol) or
                  np.isclose(sides[1], sides[2], rtol=rtol, atol=atol)):
                shape = "tri-iso"
            else:
                shape = "tri-scal"
    else:
        coords = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        centered = coords - coords.mean(axis=0, keepdims=True)
        rank = np.linalg.matrix_rank(centered, tol=1e-9)
        if rank <= 1:
            shape = "line"
        elif rank == 2:
            radii = np.linalg.norm(centered, axis=1)
            # ring if all radii identical within loose tol
            shape = "ring" if np.allclose(radii, radii[0], rtol=1e-2, atol=1e-2) else "planar"
        else:
            shape = "3D"

    # ----- field orientation (e.g., Exkz) -----
    try:
        orient = f"E{axis_label(efield.e_hat)}k{axis_label(efield.k_hat)}"
    except Exception:
        orient = "Eunknown"

    # ----- assemble name -----
    if (Np in (2, 3)) and equal_D and ((Np == 2 and shape == "line") or (Np == 3 and shape == "tri-equil")):
        # Your preferred symmetric naming
        base = f"{nname}_{Dpart}_{Gpart}_{orient}"
    else:
        base = f"{nname}-{shape}_{Dpart}_{Gpart}_{orient}"

    if lmax is not None:
        base += f"_lmax{int(lmax)}"
    if eps_h is not None:
        # compact float repr without trailing zeros if possible
        eps_str = f"{eps_h:g}"
        base += f"_epsh{eps_str}"
    if extra_tag:
        base += f"_{extra_tag}"

    if include_timestamp:
        from datetime import datetime, timezone
        base += "_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    outdir = Path(prefix) / base
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def flip_coefs_along_wavelength(lam, coef, flip_lam_too=False):
    c = np.asarray(coef)
    if c.ndim != 2:
        raise ValueError("coef must be 2-D")
    if c.shape[1] == lam.size:      # (modes, λ)
        c = c[:, ::-1]
    elif c.shape[0] == lam.size:    # (λ, modes)
        c = c[::-1, :]
    else:
        raise ValueError("coef shape doesn't match lam length")
    lam_out = lam[::-1] if flip_lam_too else lam
    return lam_out, c

def coefficients_plot(Np, lam, coef, flip=True, flip_lam_too=False):
    # ensure shape (λ, modes) for consistent plotting
    Yre, Yim = np.asarray(coef.real), np.asarray(coef.imag)
    if Yre.shape[0] != lam.size:
        Yre, Yim = Yre.T, Yim.T

    if flip:
        lam, Yre = flip_coefs_along_wavelength(lam, Yre, flip_lam_too)
        _,   Yim = flip_coefs_along_wavelength(lam, Yim, flip_lam_too=False)

    # highlight first mode; plot all others without legend
    plt.plot(lam, Yre[:, 0], 'k.-', label='real part of EM coefficients')
    plt.plot(lam, Yim[:, 0], 'k-',  label='imaginary part of EM coefficients')
    plt.plot(lam, Yre, '.-', alpha=1, label='_nolegend_')
    plt.plot(lam, Yim, '-',  alpha=1, label='_nolegend_')

    plt.title(f"Nanoparticle {Np+1} — EM coefficients vs. wavelength")
    plt.legend(); plt.tight_layout(); plt.show()


def print_active_lm(BCM, thr=1.0):
    # |coef| as (n_modes, n_lambda)
    mag = np.nan_to_num(np.abs(BCM.BCM_coef).astype(float))
    if mag.shape[1] != BCM.lam_um.size:  # fix orientation if needed
        mag = mag.T

    k = np.flatnonzero(mag.max(axis=1) > thr)   # active mode indices
    if k.size == 0:
        print("No active modes."); return

    # vectorized k -> (l, m)
    l = (np.floor(np.sqrt(k + 1) - 1) + 1).astype(int)
    m = k - ((l - 1) * (l + 1)) - l

    print(*[f"Active l,m -> l={ll}, m={mm}" for ll, mm in zip(l, m)], sep="\n")

def lorentzian_weight(hv, hv0, Gamma_exc_eV):
    """Unit-area Lorentzian in hv (eV). Gamma_exc_eV is FWHM."""
    g2 = 0.5 * Gamma_exc_eV
    return (1/np.pi) * g2 / ((hv - hv0)**2 + g2**2)

def omega_average_spectra(
    hv0,
    Gamma_exc_eV,
    compute_spectrum_at_hv,
    hv_span_factor=3.0,
    nhv=31,
    E_ref=None,
    interp_kind="linear",
):
    """
    Compute excitation-averaged hot-carrier spectra around hv0.

    Parameters
    ----------
    hv0 : float
        Central excitation energy (eV).
    Gamma_exc_eV : float
        Excitation bandwidth FWHM (eV) for Lorentzian S(hv;hv0).
    compute_spectrum_at_hv : callable
        Function: hv -> (E_grid, Ge(E), Gh(E))
        IMPORTANT: E_grid should be in eV relative to EF, consistent across runs.
    hv_span_factor : float
        Integrate over hv in [hv0 - span, hv0 + span] where span = hv_span_factor * Gamma_exc_eV.
        3*Gamma is usually enough for Lorentzian tails in practice.
    nhv : int
        Number of hv samples.
    E_ref : array or None
        Common energy grid for output. If None, uses the E_grid from the first hv.
    interp_kind : str
        Currently only linear interpolation via np.interp (fast and stable).

    Returns
    -------
    E_ref, Ge_avg, Gh_avg, hv_grid, w_norm
    """
    span = hv_span_factor * Gamma_exc_eV
    hv_grid = np.linspace(hv0 - span, hv0 + span, nhv)

    # raw weights (Lorentzian)
    w = lorentzian_weight(hv_grid, hv0, Gamma_exc_eV)

    # If hv grid uniform, normalize directly; if you later use nonuniform, include dhv factors.
    w_norm = w / np.sum(w)

    # Evaluate first spectrum to set reference energy grid
    E0, Ge0, Gh0 = compute_spectrum_at_hv(hv_grid[0])
    if E_ref is None:
        E_ref = E0

    # Helper: interpolate onto E_ref
    def to_Eref(E, y):
        if np.array_equal(E, E_ref):
            return y
        # np.interp requires ascending x
        return np.interp(E_ref, E, y, left=0.0, right=0.0)

    Ge_acc = np.zeros_like(E_ref, dtype=float)
    Gh_acc = np.zeros_like(E_ref, dtype=float)

    # accumulate
    for wk, hv in zip(w_norm, hv_grid):
        E, Ge, Gh = compute_spectrum_at_hv(hv)
        Ge_acc += wk * to_Eref(E, Ge)
        Gh_acc += wk * to_Eref(E, Gh)

    return E_ref, Ge_acc, Gh_acc, hv_grid, w_norm


# def convert_raw_hot(Te, Th, E, dE_factor=1.0, nplot=1000, renormalize=True):
#     """
#     Convert discrete (Te, Th) sampled at irregular energies E into smooth
#     energy-resolved distributions on a uniform grid using Lorentzian broadening.

#     Implements:
#         T_plot(Ej) = sum_i T(Ei) * w_i * L(Ej - Ei; Gamma)
#     where w_i are trapezoidal quadrature weights for nonuniform Ei, and
#     L is a unit-area Lorentzian with FWHM Gamma.

#     Parameters
#     ----------
#     Te, Th : array_like
#         Discrete spectra values at energies E (same length as E).
#     E : array_like
#         Energies (eV). Can be unsorted and nonuniform.
#     dE_factor : float
#         Controls Lorentzian width relative to output grid spacing:
#             Gamma = dE_factor * (E_plot[1] - E_plot[0])
#         (Gamma is FWHM, matching your lorentz_unit definition.)
#     nplot : int
#         Number of points in output uniform grid.
#     renormalize : bool
#         If True, rescales the smoothed curves so that
#             ∫ T_plot(E) dE == Σ T(Ei) w_i
#         (helps remove small edge/grid truncation errors).

#     Returns
#     -------
#     E_plot : (nplot,) ndarray
#         Uniform energy grid spanning [min(E), max(E)].
#     Te_plot, Th_plot : (nplot,) ndarray
#         Smoothed spectra on E_plot (same units as Te/Th per eV if Te/Th were per state).
#     """
#     Te = np.asarray(Te, dtype=float).ravel()
#     Th = np.asarray(Th, dtype=float).ravel()
#     E  = np.asarray(E,  dtype=float).ravel()

#     # basic sanity / finite mask
#     m = np.isfinite(E) & np.isfinite(Te) & np.isfinite(Th)
#     E, Te, Th = E[m], Te[m], Th[m]

#     if E.size < 2:
#         E_plot = np.linspace(np.nanmin(E), np.nanmax(E), nplot) if E.size == 1 else np.linspace(0, 1, nplot)
#         return E_plot, np.zeros_like(E_plot), np.zeros_like(E_plot)

#     # sort by energy
#     idx = np.argsort(E)
#     E, Te, Th = E[idx], Te[idx], Th[idx]

#     # uniform output grid
#     E_plot = np.linspace(E.min(), E.max(), int(nplot))
#     dE_plot = E_plot[1] - E_plot[0]

#     # Lorentzian FWHM (Gamma)
#     Gamma = float(dE_factor) * float(dE_plot)
#     g2 = 0.5 * Gamma  # half-width at half-maximum

#     # unit-area Lorentzian kernel L(x;Gamma)
#     def L(x):
#         return (1.0 / np.pi) * g2 / (x*x + g2*g2)

#     # trapezoid weights for nonuniform E (approx dE around each point)
#     w = np.empty_like(E)
#     w[1:-1] = 0.5 * (E[2:] - E[:-2])
#     w[0]    = (E[1] - E[0])
#     w[-1]   = (E[-1] - E[-2])

#     # compute smoothed curves: sum_i T(Ei) * w_i * L(Ej - Ei)
#     # vectorized with broadcasting (nplot x nE)
#     K = L(E_plot[:, None] - E[None, :])            # (nplot, nE)
#     Te_plot = (K * (Te * w)[None, :]).sum(axis=1)  # (nplot,)
#     Th_plot = (K * (Th * w)[None, :]).sum(axis=1)  # (nplot,)

#     if renormalize:
#         # enforce area conservation on the finite E_plot window/grid
#         target_e = np.sum(Te * w)
#         target_h = np.sum(Th * w)
#         area_e = np.trapz(Te_plot, E_plot)
#         area_h = np.trapz(Th_plot, E_plot)

#         if area_e != 0.0 and np.isfinite(area_e):
#             Te_plot *= (target_e / area_e)
#         if area_h != 0.0 and np.isfinite(area_h):
#             Th_plot *= (target_h / area_h)

#     return E_plot, Te_plot, Th_plot



def convert_raw_hot(Te, Th, E, dE_factor):
    idx = np.argsort(E)
    E = E[idx]; Te = Te[idx]; Th = Th[idx]
    E_plot = np.linspace(E.min(), E.max(), 1000)
    dE = (E_plot[1] - E_plot[0]) * dE_factor
    Phi = lambda x: (1/np.pi) * (dE/2) / (x**2 + (dE/2)**2)  # Lorentzian (unit area)

    EE, EEp = np.meshgrid(E, E_plot)
    TTe = np.meshgrid(Te, E_plot)[0]
    TTh = np.meshgrid(Th, E_plot)[0]

    Te_plot = np.sum(TTe * Phi(EEp - EE), axis=1)
    Th_plot = np.sum(TTh * Phi(EEp - EE), axis=1)

    return E_plot, Te_plot, Th_plot

# def convert_raw_hot(Te, Th, E, dE_factor):
#     idx = np.argsort(E)
#     E = E[idx]; Te = Te[idx]; Th = Th[idx]
#     E_plot = np.linspace(E.min(), E.max(), 1000)
#     dE = (E_plot[1] - E_plot[0]) * dE_factor
#     Phi = lambda x: (1/np.pi) * (dE/2) / (x**2 + (dE/2)**2)  # Lorentzian (unit area)

#     EE, EEp = np.meshgrid(E, E_plot)
#     TTe = np.meshgrid(Te, E_plot)[0]
#     TTh = np.meshgrid(Th, E_plot)[0]

#     # --- bin widths (trapezoidal) ---
#     w = np.empty_like(E)
#     if E.size > 1:
#         w[1:-1] = 0.5 * (E[2:] - E[:-2])
#         w[0]    = E[1] - E[0]
#         w[-1]   = E[-1] - E[-2]
#     else:
#         w[:] = 0.0

#     # weighted sum ≡ trapz along E
#     Te_plot = np.sum((TTe * w[None, :]) * Phi(EEp - EE), axis=1)
#     Th_plot = np.sum((TTh * w[None, :]) * Phi(EEp - EE), axis=1)
#     return E_plot, Te_plot, Th_plot


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def hot_carriers_plot(Te, Th, Te_raw, Th_raw, 
                      e_states, Np, peak, tau_e, D, hv, EF, dE_factor, delta, efield,
                      out_path):
    """
    Displays and saves the hot-carriers plot.
    - If `out_path` is None or empty, a default filename is created.
    - Returns nothing (shows the plot and saves it).
    """
    # Units/arrays
    Te_raw = Te_raw
    Th_raw = Th_raw

    # energies in SAME order as Te/Th
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    assert len(E_all) == len(Te) == len(Th), "Length mismatch between E_all and Te/Th."

    # smooth to fine grid
    x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)

    # fs^-1 → ps^-1, then per eV
    to_ps = 1000.0
    scale = to_ps / hv
    Te_x *= scale
    Th_x *= scale

    # masks
    mask_e = (x >= EF) & (x <= EF + delta)
    mask_h = (x <= EF) & (x >= EF - delta)

    # plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.fill_between((x - EF)[mask_e], Te_x[mask_e], color='r', alpha=0.38)
    ax.fill_between((x - EF)[mask_h], Th_x[mask_h], color='b', alpha=0.38)

    # SECOND Y-AXIS for bars
    ax2 = ax.twinx()
    bar_width = 2.0e-2
    ax2.bar(E_all - EF, Te_raw * to_ps, width=bar_width, color='firebrick', alpha=0.9, label='Electrons')
    ax2.bar(E_all - EF, Th_raw * to_ps, width=bar_width, color='royalblue', alpha=0.9, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, color='gray', alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, color='gray', alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_xlabel('Hot carrier energy relative to Fermi level (eV)')
    ax.set_ylabel(r'HC generation rate density')
    # ax.set_ylabel(r'HC generation rate density $[{10}^{-3}\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax2.set_ylabel('HC generation rate per particle $[\mathrm{ps}^{-1}]$')

    # y-limits
    ax.set_ylim(0, 1.05 * max(Te_x[mask_e].max(initial=0), Th_x[mask_h].max(initial=0)))
    ax2.set_ylim(0, 3.0 * to_ps * max(Te_raw.max(initial=0), Th_raw.max(initial=0)))

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right')

    tau_fs = float(np.atleast_1d(tau_e)[0])
    ax.set_title(rf'Nanoparticle N°{Np}, Resonance peak N°{peak}, D = {D} nm, '
                 rf'$\tau = ${tau_fs/1000:.2f} ps, $h\nu$ = {hv:.3f} eV')
    ax.grid(True, ls=':')
    plt.tight_layout() 

    # -------- Save to out_path (create folders if needed) --------
    if out_path is None or str(out_path).strip() == "":
        out_path = path_with_polarization(f'hot_carriers_N{Np}_peak{peak}_D{D}nm_tau{int(tau_fs)}fs_hv{hv:.2f}eV.png', efield)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",transparent=True)

    # Show and close (no return)
    plt.show()
    plt.close(fig)




def hot_carrier_dynamics_plot(Te, Th, Te_raw, Th_raw, 
                              e_states, Np, peak, tau_e, D, hv, EF, dE_factor, delta, fps, out_path, bar_width=2.0e-2):

    # energies in SAME order as Te/Th
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real

    # arrays & sanity
    Te_arr   = np.asarray(Te,     float)/1000  # (n_tau, N)
    Th_arr   = np.asarray(Th,     float)/1000
    Te_raw_a = np.asarray(Te_raw, float)
    Th_raw_a = np.asarray(Th_raw, float)
    tau_arr  = np.asarray(tau_e,  float)

    n_tau, N = Te_arr.shape
    assert Th_arr.shape   == (n_tau, N)
    assert Te_raw_a.shape == (n_tau, N)
    assert Th_raw_a.shape == (n_tau, N)
    assert E_all.size     == N

    # fixed y-limits
    to_ps = 1000.0  # fs -> ps
    def frame_max_lines(i):
        x, Te_x, Th_x = convert_raw_hot(Te_arr[i], Th_arr[i], E_all, dE_factor)
        s = to_ps / hv
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)
        m_e = Te_x[mask_e].max() if np.any(mask_e) else 0.0
        m_h = Th_x[mask_h].max() if np.any(mask_h) else 0.0
        return max(m_e, m_h)

    YMAX  = 1.05 * max(frame_max_lines(i) for i in range(n_tau)) if n_tau else 1.0
    YMAX2 = 1.20 * to_ps * max(np.max(Te_raw_a), np.max(Th_raw_a)) if n_tau else 1.0

    # figure & artists
    fig, ax = plt.subplots(figsize=(20, 4.5))
    line_e, = ax.plot([], [], lw=1.8, color='r', alpha = 0.5)
    line_h, = ax.plot([], [], lw=1.8, color='b', alpha = 0.5)

    # will hold the fill_between polygons; start empty
    fill_e = None
    fill_h = None

    # secondary axis for bars
    ax2 = ax.twinx()
    x_bar = E_all - EF
    bars_e = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width,
                     color='firebrick', alpha=1, label='Electrons')
    bars_h = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width,
                     color='royalblue',  alpha=1, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_ylim(0, YMAX)
    ax2.set_ylim(0, YMAX2)

    ax.set_xlabel('Hot carrier energy relative to $E_F$ (eV)')
    ax.set_ylabel(r'Hot carrier generation rate density $[{10}^{-3}\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax2.set_ylabel('hot carrier generation rate per particle $[\mathrm{ps}^{-1}]$')  # ← label for the bars’ axis
    ax.grid(True, ls=':')

    # merged legend (lines + bars; fills get no legend via '_nolegend_')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right')

    # frame builder
    def make_frame_data(i):
        x, Te_x, Th_x = convert_raw_hot(Te_arr[i], Th_arr[i], E_all, dE_factor)
        s = to_ps / hv
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)
        return (x - EF)[mask_e], Te_x[mask_e], (x - EF)[mask_h], Th_x[mask_h]

    def init():
        nonlocal fill_e, fill_h
        line_e.set_data([], [])
        line_h.set_data([], [])
        if fill_e: fill_e.remove(); fill_e = None
        if fill_h: fill_h.remove(); fill_h = None
        for r in list(bars_e) + list(bars_h):
            r.set_height(0.0)
        ax.set_title('')
        return [line_e, line_h, *bars_e.patches, *bars_h.patches]

    def update(i):
        nonlocal fill_e, fill_h
        xe, ye, xh, yh = make_frame_data(i)

        # lines (for legend/edges)
        line_e.set_data(xe, ye)
        line_h.set_data(xh, yh)

        # refresh fills each frame
        if fill_e: fill_e.remove()
        fill_e = ax.fill_between(xe, 0.0, ye, color='r', alpha=0.3, label='_nolegend_')
        if fill_h: fill_h.remove()
        fill_h = ax.fill_between(xh, 0.0, yh, color='b', alpha=0.3, label='_nolegend_')

        # bars (secondary axis)
        he = to_ps * Te_raw_a[i]
        hh = to_ps * Th_raw_a[i]
        for rect, h in zip(bars_e, he): rect.set_height(h)
        for rect, h in zip(bars_h, hh): rect.set_height(h)

        tau_ps = tau_arr[i] / 1000.0
    
        ax.set_title(rf'Nanoparticle N°{Np}, Resonance peak N°{peak}, D = {D} nm, $\tau$ = {tau_ps:.2f} ps, $h\nu$ = {hv:.2f} eV   (frame {i+1}/{n_tau})')
        return [line_e, line_h, *bars_e.patches, *bars_h.patches]

    anim = FuncAnimation(fig, update, frames=n_tau, init_func=init,
                         blit=False, interval=int(1000 / fps))
    writer = PillowWriter(fps=fps, metadata={'artist': 'Simulation'})
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

        # (Optional) if running in a Jupyter notebook, show it inline:
    try:
        from IPython.display import Image, display, HTML
        display(HTML(f"<b>Saved GIF:</b> {out_path}"))
        display(Image(filename=out_path))
    except Exception:
        pass

def _edges_from_centers(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mid = 0.5*(x[1:] + x[:-1])
    first = x[0] - (mid[0] - x[0])
    last  = x[-1] + (x[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))

def _collapse_equal_energies(M_ord: np.ndarray, E_ord: np.ndarray, how: str = "sum", tol: float = 0.0):
    N = E_ord.size
    jump = (np.diff(E_ord) > tol) if tol > 0.0 else (np.diff(E_ord) != 0.0)
    starts = np.concatenate(([0], np.where(jump)[0] + 1))
    counts = np.diff(np.concatenate((starts, [N])))
    E_unique = E_ord[starts]
    R = np.add.reduceat(M_ord, starts, axis=0)       # (Nu, N)
    M_coll = np.add.reduceat(R, starts, axis=1)      # (Nu, Nu)
    if how == "mean":
        c = counts.astype(float)
        M_coll = M_coll / (c[:, None] * c[None, :])
    elif how != "sum":
        raise ValueError("how must be 'sum' or 'mean'.")
    return M_coll, E_unique, counts

def plot_transition_matrix_colormap(
    Mfi2: np.ndarray,
    E_all: np.ndarray,
    *,
    relative_to_EF: bool = False,
    E_F: float = 0.0,
    dedup: str = "sum",       # 'sum', 'mean', or 'none'
    tol: float = 0.0,         # energy equality tolerance (eV) for dedup
    # ---- NEW SCALING CONTROLS ----
    scale: str = "log",     # 'asinh' (default), 'pow', 'linear', 'log'
    gamma: float = 0.5,       # used when scale='pow' (0.4–0.7 is a nice range)
    tiny: float = 1e-30,      # used when scale='log'
    qmin: float = 0.001,      # percentile clip (0..1)
    qmax: float = 0.999,
    soft: float | None = None,# used when scale='asinh'; if None, auto
    xlabel: str = "Final-state energy (eV)",
    ylabel: str = "Initial-state energy (eV)",
    title: str | None = None,
    savepath: str | None = None,
) -> None:
    """Energy-sorted heatmap with gentle contrast options."""
    M = np.asarray(Mfi2, float)
    E = np.asarray(E_all, float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Mfi2 must be a square (N,N).")
    if E.ndim != 1 or E.size != M.shape[0]:
        raise ValueError("E_all must be (N,) matching Mfi2.")

    # Sort by energy
    E_plot = E - E_F if relative_to_EF else E.copy()
    order  = np.argsort(E_plot)
    M_ord  = M[order][:, order]
    E_ord  = E_plot[order]

    # Collapse degeneracies for pcolormesh
    if dedup != "none":
        M_plot, E_plot_u, _ = _collapse_equal_energies(M_ord, E_ord, how=dedup, tol=tol)
    else:
        if np.any(np.diff(E_ord) == 0.0):
            # fallback to imshow if exact duplicates and no dedup requested
            Zraw = M_ord
            vmin = np.quantile(Zraw, qmin)
            vmax = np.quantile(Zraw, qmax)
            Zraw = np.clip(Zraw, vmin, vmax)
            if scale == "log":
                Z = np.log10(Zraw + tiny)
                cbl = r"log$_{10}(M_{fi}^2)$"
            elif scale == "pow":
                Z = np.power(Zraw / vmax, gamma)
                cbl = rf"$(M_{{fi}}^2)^\gamma$ (γ={gamma:.2f})"
            elif scale == "asinh":
                s = soft
                pos = Zraw[Zraw > 0]
                if s is None:
                    s = (np.percentile(pos, 95)/3.0) if pos.size else 1.0
                Z = np.arcsinh(Zraw / s)
                cbl = r"asinh($M_{fi}^2 / s$)"
            else:
                Z = Zraw
                cbl = r"$M_{fi}^2$"
            fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
            im = ax.imshow(Z, origin="lower", aspect="equal",
                           extent=[E_ord[0], E_ord[-1], E_ord[0], E_ord[-1]])
            cb = fig.colorbar(im, ax=ax); cb.set_label(cbl)
            ax.set_xlabel(xlabel if not relative_to_EF else xlabel.replace("(eV)", "− E$_F$ (eV)"))
            ax.set_ylabel(ylabel if not relative_to_EF else ylabel.replace("(eV)", "− E$_F$ (eV)"))
            if title: ax.set_title(title)
            if savepath: fig.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.show()
            return
        else:
            M_plot, E_plot_u = M_ord, E_ord

    import matplotlib.colors as mcolors

    pos  = M_plot[M_plot > 0]
    vmax = np.quantile(M_plot, qmax) if pos.size else 1.0
    lmin = max(np.percentile(pos, 1) if pos.size else tiny, vmax * 1e-3)

    Z    = np.ma.masked_where(M_plot <= 0, np.clip(M_plot, lmin, vmax))
    norm = mcolors.LogNorm(vmin=lmin, vmax=vmax)
    cbl  = r"$|M_{fi}|^2$"

    edges = _edges_from_centers(E_plot_u)
    cmap  = plt.get_cmap("hot").copy()
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.set_facecolor("black")
    mesh = ax.pcolormesh(edges, edges, Z, shading="auto", norm=norm, cmap=cmap)
    cb = fig.colorbar(mesh, ax=ax); cb.set_label(cbl)
    ax.set_xlabel(xlabel if not relative_to_EF else xlabel.replace("(eV)", "− E$_F$ (eV)"))
    ax.set_ylabel(ylabel if not relative_to_EF else ylabel.replace("(eV)", "− E$_F$ (eV)"))
    if title: ax.set_title(title)
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(edges[0], edges[-1])
    ax.set_aspect("equal", adjustable="box")
    if savepath: fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_mfi_abs(
    Mfi2: np.ndarray,
    E_vals: np.ndarray,
    *,
    dE_lines: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
    decades: float = 3.0,
) -> None:
    """Ei/Ef heatmap of |Mfi| with LogNorm + line plot vs Ef-Ei."""
    import matplotlib.colors as mcolors

    M = np.sqrt(np.asarray(Mfi2, float))
    EI, EF_grid = np.meshgrid(E_vals, E_vals)

    pos  = M[M > 0]
    vmax = np.percentile(pos, 99)
    lmin = np.percentile(pos, 20)
    Z    = np.ma.masked_where(M <= 0, np.clip(M, lmin, vmax))
    norm = mcolors.LogNorm(vmin=lmin, vmax=vmax)
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("black")

    _colors = plt.cm.tab10(np.linspace(0, 1, len(dE_lines)))
    x       = (EF_grid - EI).ravel()
    order   = np.argsort(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_facecolor("black")
    sc = axes[0].pcolormesh(EI, EF_grid, Z, cmap=cmap, norm=norm)
    plt.colorbar(sc, ax=axes[0], label='$|M_{fi}|$')
    for dE, col in zip(dE_lines, _colors):
        axes[0].plot(E_vals, E_vals + dE, lw=1.2, ls='--', color=col, label=f'$E_f-E_i={dE}$')
    axes[0].set_xlim(E_vals.min(), E_vals.max())
    axes[0].set_ylim(E_vals.min(), E_vals.max())
    axes[0].set_xlabel('$E_i$ (eV)', fontsize=12)
    axes[0].set_ylabel('$E_f$ (eV)', fontsize=12)
    axes[0].set_title('$|M_{fi}|$', fontsize=13)
    axes[0].legend(fontsize=7)

    axes[1].plot(x[order], M.ravel()[order], lw=1.5, color='seagreen')
    for dE, col in zip(dE_lines, _colors):
        axes[1].axvline(dE, lw=1.2, ls='--', color=col)
    axes[1].set_xlabel('$E_f - E_i$ (eV)', fontsize=12)
    axes[1].set_ylabel('$|M_{fi}|$', fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_Ne_cdf_steps(
    eps_axis: np.ndarray,
    Ne_clean: np.ndarray,     # shape (Ntau, Nε)
    tau_e: np.ndarray,        # fs, length Ntau
    *,
    hv: float | None = None,                 # eV
    E_lines: list[float] | None = None,      # absolute energies (eV) for vertical refs
    E_line_factors: tuple[float, float] | None = (0.2, 0.5),
    figsize: tuple[float, float] = (5.6, 3.6),
    dpi: int = 140,
    cmap_name: str = "viridis",
    xlabel: str = r"$\varepsilon = E_f - E_F$ (eV)",
    ylabel: str = r"$N_e$ per absorbed photon",
    # --- new bits ---
    Np: int | None = None,
    peak: int | None = None,
    D: float | None = None,
    efield=None,                               # pass your EField here
    out_dir: str | Path | None = None,         # optional directory override
    save_dpi: int = 300,
    save_transparent: bool = False,
    ax: plt.Axes | None = None,
    close: bool = False,
) -> None:
    """
    Plots Ne(ε) steps colored by τ, saves to a name that includes
    hot_carriers_N{Np}_peak{peak}_D{D}nm_tau{int(tau_fs)}fs_hv{hv:.2f}eV
    and appends the polarization tag from `efield` (e.g., _Ey).
    """
    eps_axis = np.asarray(eps_axis, float)
    Ne_clean = np.asarray(Ne_clean, float)
    tau_e = np.asarray(tau_e, float)

    if Ne_clean.ndim != 2:
        raise ValueError("Ne_clean must be 2D with shape (Ntau, Nε).")
    if Ne_clean.shape[1] != eps_axis.size:
        raise ValueError("Ne_clean.shape[1] must equal eps_axis.size.")
    if Ne_clean.shape[0] != tau_e.size:
        raise ValueError("len(tau_e) must equal Ne_clean.shape[0].")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # Colormap by tau
    cmap = plt.get_cmap(cmap_name)
    tmin = float(np.min(tau_e)); tmax = float(np.max(tau_e))
    if np.isclose(tmin, tmax): tmin, tmax = tmin - 1e-12, tmax + 1e-12
    norm = plt.Normalize(vmin=tmin, vmax=tmax)

    for i, tau in enumerate(tau_e):
        ax.step(eps_axis, Ne_clean[i], where="post",
                lw=1.0, alpha=0.8, color=cmap(norm(float(tau))))

    # Vertical reference lines
    lines = list(E_lines) if E_lines is not None else (
        [float(f)*float(hv) for f in E_line_factors] if (hv is not None and E_line_factors is not None) else None
    )
    ymax = float(Ne_clean.max()) * 1.05
    if lines:
        for Eref in lines:
            ax.axvline(Eref, ls="--", lw=1.2, color="k", alpha=0.7, zorder=3)
            ax.text(Eref, 0.98*ymax, f"{Eref:.1f} eV",
                    ha="right", va="top", rotation=90, fontsize=9, color="k", alpha=0.8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01); cbar.set_label("τ (fs)")

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, float(eps_axis.max())); ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ---- Build the requested out_path and append polarization ----
    if any(v is None for v in (Np, peak, D, hv)):
        base_name = "hot_carriers"
    else:
        base_name = f"HC_efficiency_N{Np+1}_peak{peak}_D{D}nm_hv{hv:.2f}eV.png"

    # append polarization (e_hat) to the filename
    save_name = path_with_polarization(base_name, efield) if efield is not None else Path(base_name)

    # choose directory
    out_dir = Path("." if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = (out_dir / save_name.name) if out_dir.is_dir() else out_dir

    fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", transparent=save_transparent)

    if created_fig:
        plt.show()
        if close:
            plt.close(fig)

def hot_carrier_gap_dynamics_gif(
    Te_tau_list, Th_tau_list, Te_raw_tau_list, Th_raw_tau_list,
    e_states,
    *,
    gap_values_nm,        # sequence of Δ values (same order as lists)
    hv_list,              # sequence of photon energies (eV), one per Δ frame
    EF: float,
    D: float,
    tau_fs: float,        # fixed lifetime shown in the title (fs)
    dE_factor: float = 5,
    Ewin: float = 4.0,    # energy window around EF for the lines/fills (eV)
    efield=None,
    fps: int = 12,
    bar_width: float = 2.0e-2,
    out_path: str | Path | None = None,
    title_prefix: str = "",
):
    """
    Animate hot-carrier spectra across *gap* Δ (delta_) for a fixed τ.

    Parameters
    ----------
    Te_tau_list, Th_tau_list : list[np.ndarray]
        One 1D array per Δ (length = number of QW levels), for the *smoothed*
        electron/hole rates at the chosen τ. If you only have the raw arrays,
        pass them in Te_raw_tau_list/Th_raw_tau_list and let this function do
        the line smoothing (done here).
    Te_raw_tau_list, Th_raw_tau_list : list[np.ndarray]
        One 1D array per Δ (length = number of QW levels), *raw* per-level rates
        at the chosen τ (used for the bar plot on the secondary y-axis).
    e_states : QW level set (used only to reconstruct the energy grid E_all)
    gap_values_nm : array-like
        Δ values (nm) in the same order as the lists.
    hv_list : array-like
        Photon energy (eV) to use on each frame (e.g., the tracked resonance energy at that Δ).
    EF : float
        Fermi level (eV).
    D : float
        Sphere diameter (nm), for labeling only.
    tau_fs : float
        Lifetime (fs) used for these per-Δ snapshots (purely cosmetic in title).
    dE_factor : float
        Spectral smoothing factor passed to convert_raw_hot.
    Ewin : float
        Half-window (eV) shown around EF for the smoothed lines.
    efield : EField | None
        Used to append polarization tag to the default filename.
    fps : int
        GIF framerate.
    out_path : str | Path | None
        If None, a default filename with polarization tag is created.
    title_prefix : str
        Optional string prepended to the title (e.g., "struct=5 · sphere 2 · peak 1").
    """
    # ------------ sanity / prep ------------
    gaps = np.asarray(gap_values_nm, float)
    hv_list = np.asarray(hv_list, float)
    nF = len(gaps)

    if not (len(Te_tau_list) == len(Th_tau_list) == len(Te_raw_tau_list) == len(Th_raw_tau_list) == nF):
        raise ValueError("All input lists must have the same length as gap_values_nm.")

    # energies associated to the level indices
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    N = E_all.size

    # ensure arrays
    Te_tau_list   = [np.asarray(v, float).reshape(N) for v in Te_tau_list]
    Th_tau_list   = [np.asarray(v, float).reshape(N) for v in Th_tau_list]
    Te_raw_list   = [np.asarray(v, float).reshape(N) for v in Te_raw_tau_list]
    Th_raw_list   = [np.asarray(v, float).reshape(N) for v in Th_raw_tau_list]

    to_ps = 1000.0  # fs → ps

    # Pre-compute global y-limits across all frames for stable axes
    def frame_envelopes(i):
        # smooth lines for this frame
        x, Te_x, Th_x = convert_raw_hot(Te_tau_list[i], Th_tau_list[i], E_all, dE_factor)
        s = to_ps / max(hv_list[i], 1e-12)
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + Ewin)
        mask_h = (x <= EF) & (x >= EF - Ewin)
        y1 = 0.0
        if np.any(mask_e): y1 = max(y1, float(Te_x[mask_e].max()))
        if np.any(mask_h): y1 = max(y1, float(Th_x[mask_h].max()))
        y2 = max(float(Te_raw_list[i].max(initial=0)), float(Th_raw_list[i].max(initial=0))) * to_ps
        return y1, y2

    if nF == 0:
        raise ValueError("No frames to animate (gap_values_nm is empty).")

    YMAX_lines = 1.05 * max(frame_envelopes(i)[0] for i in range(nF))
    YMAX_bars  = 1.20 * max(frame_envelopes(i)[1] for i in range(nF))

    # ------------ figure / artists ------------
    fig, ax = plt.subplots(figsize=(20, 4.5))
    line_e, = ax.plot([], [], lw=1.8, color='r', alpha=0.5)
    line_h, = ax.plot([], [], lw=1.8, color='b', alpha=0.5)
    fill_e = None
    fill_h = None

    # bars on twin axis
    ax2 = ax.twinx()
    x_bar = E_all - EF
    bars_e = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width, color='firebrick', alpha=1, label='Electrons')
    bars_h = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width, color='royalblue',  alpha=1, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, alpha=0.5)
    # we draw ±hv in each frame inside update(); initial state shows nothing

    ax.set_xlim(-Ewin, +Ewin)
    ax.set_ylim(0, YMAX_lines)
    ax2.set_ylim(0, YMAX_bars)

    ax.set_xlabel(r'Hot carrier energy relative to $E_F$ (eV)')
    ax.set_ylabel(r'Hot carrier generation rate density $[{10}^{-3}\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax2.set_ylabel(r'Hot carrier generation rate per particle $[\mathrm{ps}^{-1}]$')
    ax.grid(True, ls=':')

    # legend (lines + bars)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right')

    # ------------ frame helpers ------------
    def make_frame_data(i):
        x, Te_x, Th_x = convert_raw_hot(Te_tau_list[i], Th_tau_list[i], E_all, dE_factor)
        s = to_ps / max(hv_list[i], 1e-12)
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + Ewin)
        mask_h = (x <= EF) & (x >= EF - Ewin)
        return (x - EF)[mask_e], Te_x[mask_e], (x - EF)[mask_h], Th_x[mask_h]

    # keep references to the ±hv lines so we can remove/redraw them each frame
    hv_lines = []

    def init():
        nonlocal fill_e, fill_h, hv_lines
        line_e.set_data([], [])
        line_h.set_data([], [])
        if fill_e: fill_e.remove(); fill_e = None
        if fill_h: fill_h.remove(); fill_h = None
        for r in list(bars_e) + list(bars_h):
            r.set_height(0.0)
        for ln in hv_lines:
            ln.remove()
        hv_lines = []
        ax.set_title('')
        return [line_e, line_h, *bars_e.patches, *bars_h.patches]

    def update(i):
        nonlocal fill_e, fill_h, hv_lines
        xe, ye, xh, yh = make_frame_data(i)

        # lines for legend/edges
        line_e.set_data(xe, ye)
        line_h.set_data(xh, yh)

        # refresh fills
        if fill_e: fill_e.remove()
        fill_e = ax.fill_between(xe, 0.0, ye, color='r', alpha=0.30, label='_nolegend_')
        if fill_h: fill_h.remove()
        fill_h = ax.fill_between(xh, 0.0, yh, color='b', alpha=0.30, label='_nolegend_')

        # bars
        he = to_ps * Te_raw_list[i]
        hh = to_ps * Th_raw_list[i]
        for rect, h in zip(bars_e, he): rect.set_height(float(h))
        for rect, h in zip(bars_h, hh): rect.set_height(float(h))

        # update ±hv markers
        for ln in hv_lines:
            ln.remove()
        hv_lines = []
        hv = float(hv_list[i])
        hv_lines.append(ax.axvline(+hv, ls='--', lw=1, color='gray', alpha=0.6))
        hv_lines.append(ax.axvline(-hv, ls='--', lw=1, color='gray', alpha=0.6))

        # title
        tau_ps = tau_fs / 1000.0
        prefix = (title_prefix + " · ") if title_prefix else ""
        ax.set_title(
            rf"{prefix}$\Delta$ = {gaps[i]:.2f} nm, "
            rf"$\tau$ = {tau_ps:.2f} ps, $h\nu$ = {hv:.2f} eV,  D = {D:.1f} nm  (frame {i+1}/{len(gaps)})"
        )
        return [line_e, line_h, *bars_e.patches, *bars_h.patches, *hv_lines]

    anim = FuncAnimation(fig, update, frames=nF, init_func=init, blit=False,
                         interval=int(1000 / max(fps,1)))
    # default output path
    if out_path is None or str(out_path).strip() == "":
        # use hv of first frame for the name; append polarization tag if available
        base = f"HC_gap_dynamics_D{D:.1f}nm_tau{int(tau_fs)}fs_hv{hv_list[0]:.2f}eV.gif"
        out_path = path_with_polarization(base, efield) if efield is not None else Path(base)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps, metadata={'artist': 'Simulation'})
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

    # try to preview when in notebook
    try:
        from IPython.display import Image, display, HTML
        display(HTML(f"<b>Saved GIF:</b> {out_path}"))
        display(Image(filename=out_path))
    except Exception:
        pass

def absorption_gap_gif(
    energy_eV: np.ndarray,
    Qabs_frames: list[np.ndarray],          # one Qabs(energy) array per Δ
    *,
    gap_values_nm: np.ndarray,              # same length as Qabs_frames
    hv_lines_per_delta: list[list[float]],  # list of lists of hv (eV) per frame
    title_prefix: str = "",
    fps: int = 12,
    out_path: str | Path | None = None,
    ymargin: float = 1.08,                  # top padding for y-limit
):
    """
    Animate absorption Q_abs(ℏω) across gap Δ.
    Draw vertical lines at every hv present in the current frame.

    Parameters
    ----------
    energy_eV : (M,) array of photon energies.
    Qabs_frames : list of (M,) arrays, one per Δ frame.
    gap_values_nm : (NΔ,) array of gap values (nm).
    hv_lines_per_delta : list of lists; per frame, a list of energies (eV) to mark.
    """
    energy_eV = np.asarray(energy_eV, float).ravel()
    NΔ = len(Qabs_frames)
    assert NΔ == len(gap_values_nm) == len(hv_lines_per_delta), "Frame lists must align with gap_values_nm."

    # Robust y-limit across frames
    ymax = 0.0
    for Qa in Qabs_frames:
        if Qa is None: continue
        Qa = np.asarray(Qa, float).ravel()
        if Qa.size: ymax = max(ymax, float(np.nanmax(Qa)))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    (line,) = ax.plot([], [], lw=2.0, color='k')
    ax.set_xlim(float(energy_eV.min()), float(energy_eV.max()))
    ax.set_ylim(0.0, ymargin * ymax)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Absorption efficiency $Q_{\\mathrm{abs}}$")
    ax.grid(True, ls=":")
    vlines = []  # will hold Line2D objects for hv markers

    def init():
        line.set_data([], [])
        for ln in vlines:
            try: ln.remove()
            except Exception: pass
        vlines.clear()
        ax.set_title("")
        return [line]

    def update(i):
        # update curve
        Qa = np.asarray(Qabs_frames[i], float).ravel()
        line.set_data(energy_eV, Qa)

        # redraw hv vertical lines for this frame
        for ln in vlines:
            try: ln.remove()
            except Exception: pass
        vlines.clear()
        for hv in hv_lines_per_delta[i]:
            vlines.append(ax.axvline(float(hv), ls='--', lw=1.2, color='crimson', alpha=0.85))

        ax.set_title(f"{title_prefix}  Δ = {gap_values_nm[i]:.2f} nm  (frame {i+1}/{NΔ})")
        return [line, *vlines]

    anim = FuncAnimation(fig, update, frames=NΔ, init_func=init, blit=False,
                         interval=int(1000/max(fps,1)))
    if out_path is None or str(out_path).strip() == "":
        out_path = Path("absorption_gap.gif")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps, metadata={'artist': 'Simulation'})
    anim.save(out_path, writer=writer, dpi=140)
    plt.close(fig)

    try:
        from IPython.display import Image, display, HTML
        display(HTML(f"<b>Saved GIF:</b> {out_path}"))
        display(Image(filename=out_path))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Optical-response dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def find_dominant_peak(x, y, prominence_frac=0.05):
    """
    Return the index and x-value of the highest-prominence peak in *y*.

    Parameters
    ----------
    x : array-like
        x-axis values (e.g. photon energy [eV]).
    y : array-like
        Signal to search (e.g. Q_abs).
    prominence_frac : float
        Minimum prominence threshold as a fraction of ``max(|y|)``.

    Returns
    -------
    idx : int or None
        Array index of the dominant peak, or ``None`` if no peak is found.
    x_peak : float or None
        Corresponding x-value, or ``None`` if no peak is found.
    """
    from scipy.signal import find_peaks as _find_peaks
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    threshold = prominence_frac * np.nanmax(np.abs(y))
    peaks, props = _find_peaks(y, prominence=threshold)
    if len(peaks) == 0:
        return None, None
    dominant = peaks[np.argmax(props['prominences'])]
    return int(dominant), float(x[dominant])


def label_peaks(ax, x, y, n_peaks=3, prominence_frac=0.05, color='steelblue', fontsize=8):
    """Annotate the top peaks on an axes with vertical lines and energy labels."""
    from scipy.signal import find_peaks
    threshold = prominence_frac * np.nanmax(np.abs(y))
    peaks, props = find_peaks(y, prominence=threshold)
    if len(peaks) == 0:
        return
    top = np.sort(peaks[np.argsort(props['prominences'])[::-1][:n_peaks]])
    y_span = np.nanmax(y) - np.nanmin(y)
    ax.set_ylim(top=np.nanmax(y) + 0.28 * y_span)
    for p in top:
        xp, yp = x[p], y[p]
        ax.axvline(xp, ls=':', color=color, lw=1.5, alpha=0.7)
        ax.text(xp, yp + 0.05 * y_span, f'{xp:.3f} eV',
                ha='center', va='bottom', fontsize=fontsize, color=color)


def plot_optical_dashboard(results, *, config=None, show=True):
    """
    Plot and save the optical-response dashboard from run_optical_response output.

    Parameters
    ----------
    results : dict
        Output of ``bcm.run_optical_response(config)``.
    config : dict, optional
        Builder config (for cluster description in title).
        If None, a generic title is used.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    None  (saves dashboard + individual spectra + geometry.txt to results['outdir'])
    """
    import math, textwrap
    from collections import Counter
    from pathlib import Path

    BCM_objects = results['BCM_objects']
    efield      = results['efield']
    E_eV        = results['E_eV']
    Pabs        = results['Pabs']
    I0          = results['I0']
    Qsca_total  = results['Qsca_total']
    Qabs_total  = results['Qabs_total']
    Qext_total  = results['Qext_total']
    eps_h       = results['eps_h']
    outdir      = Path(results['outdir'])
    Np          = len(BCM_objects)

    # ── Title ──
    field_tag = f'E = {axis_label(efield.e_hat)},  k = {axis_label(efield.k_hat)}'
    if config is not None:
        groups = Counter(zip(config['materials'], config['diameters']))
        cluster_tag = ',  '.join(
            f'{"" if n == 1 else f"{n}× "}{mat} D={D:.1f} nm'
            for (mat, D), n in groups.items()
        )
    else:
        cluster_tag = f'{Np} particles'
    cluster_lines = textwrap.wrap(cluster_tag, width=55)

    # ── Dashboard ──
    n_cols = 3
    n_rows = 1 + math.ceil(Np / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    suptitle_text = 'Optical Response\n' + field_tag + '\n' + '\n'.join(cluster_lines)
    fig.suptitle(suptitle_text, fontsize=12, fontweight='bold', linespacing=1.6)

    # Row 0: totals
    for ax, data, color, title, ylabel in [
        (axes[0, 0], Qsca_total, 'C0', 'Total Scattering Efficiency', r'$Q_{sca}$'),
        (axes[0, 1], Qabs_total, 'k',  'Total Absorption Efficiency', r'$Q_{abs}$'),
        (axes[0, 2], Qext_total, 'C3', 'Total Extinction Efficiency', r'$Q_{ext}$'),
    ]:
        ax.plot(E_eV, data, color=color, lw=2)
        label_peaks(ax, E_eV, data, color=color if color != 'k' else 'dimgray')
        ax.set_title(title)
        ax.set_xlabel('Photon energy (eV)'); ax.set_ylabel(ylabel); ax.grid(True, ls=':')

    # Rows 1+: per-sphere absorption
    for i, obj in enumerate(BCM_objects, start=1):
        row = 1 + (i - 1) // n_cols
        col = (i - 1) % n_cols
        ax = axes[row, col]
        Qabs_i = Pabs[i-1] / (I0 * np.pi * (obj.diameter / 2.0)**2)
        ax.plot(E_eV, Qabs_i, 'C1', lw=2)
        label_peaks(ax, E_eV, Qabs_i, color='C1')
        ax.set_title(
            f'{obj.label}  |  D = {obj.diameter:.1f} nm\n'
            f'pos = ({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f}) nm'
        )
        ax.set_xlabel('Photon energy (eV)'); ax.set_ylabel(r'$Q_{abs}$'); ax.grid(True, ls=':')

    for r in range(1, n_rows):
        for c in range(n_cols):
            if (r - 1) * n_cols + c >= Np:
                axes[r, c].set_visible(False)

    n_title_lines = 2 + len(cluster_lines)
    top_frac = max(0.78, 1.0 - 0.045 * n_title_lines)
    fig.tight_layout(rect=[0, 0, 1, top_frac])
    fig.savefig(outdir / "optical_response.png", dpi=200, bbox_inches='tight', transparent=True)
    if show:
        plt.show()
    plt.close(fig)

    # ── Individual spectrum files ──
    for i, obj in enumerate(BCM_objects, start=1):
        fig_i, ax_i = plt.subplots(figsize=(6, 4))
        Qabs_i = Pabs[i-1] / (I0 * np.pi * (obj.diameter / 2.0)**2)
        ax_i.plot(E_eV, Qabs_i, 'C1')
        label_peaks(ax_i, E_eV, Qabs_i, color='C1')
        ax_i.set_xlabel('Photon energy (eV)'); ax_i.set_ylabel(r'$Q_{abs}$'); ax_i.grid(True, ls=':')
        fig_i.tight_layout()
        fig_i.savefig(outdir / f"absorption_spectrum_sphere{i}.png", dpi=200, transparent=True)
        plt.close(fig_i)

    for fname, data, color in [
        ("scattering_spectrum_total.png", Qsca_total, 'C0'),
        ("absorption_spectrum_total.png", Qabs_total, 'k'),
        ("extinction_spectrum_total.png", Qext_total, 'C3'),
    ]:
        fig_i, ax_i = plt.subplots(figsize=(6, 4))
        ax_i.plot(E_eV, data, color=color)
        label_peaks(ax_i, E_eV, data, color=color if color != 'k' else 'dimgray')
        ax_i.set_xlabel('Photon energy (eV)'); ax_i.grid(True, ls=':')
        fig_i.tight_layout()
        fig_i.savefig(outdir / fname, dpi=200, transparent=True)
        plt.close(fig_i)

    # ── Geometry summary (.xyz — extended XYZ, readable by ASE / OVITO) ──
    e = efield.e_hat;  k = efield.k_hat
    comment = (
        f"eps_h={eps_h} "
        f"E0={efield.E0}_V/nm "
        f"e_hat=\"{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}\" "
        f"k_hat=\"{k[0]:.4f} {k[1]:.4f} {k[2]:.4f}\" "
        f"Properties=species:S:1:pos:R:3:diameter:R:1:lmax:I:1"
    )
    with open(outdir / "geometry.xyz", "w") as f:
        f.write(f"{Np}\n{comment}\n")
        for obj in BCM_objects:
            symbol = obj.label.split()[0]          # first word (e.g. "Au", "Ag", "sphere1")
            p = obj.position
            f.write(f"{symbol}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}"
                    f"  {obj.diameter:.3f}  {obj.lmax}\n")

    print("All figures and data saved in:", outdir)


# ═══════════════════════════════════════════════════════════════════════════════
# E-field map computation and plotting
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
from plytrons.math_utils import eps0, em_sph_harm


def _lm_arrays(lmax: int):
    li, mi = [], []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            li.append(l); mi.append(m)
    return np.asarray(li, dtype=np.int64), np.asarray(mi, dtype=np.int64)


def _cart2sph(x, y, z):
    r     = np.sqrt(x*x + y*y + z*z)
    theta = np.arctan2(np.sqrt(x*x + y*y), z)
    phi   = np.arctan2(y, x)
    return r, theta, phi


def phi_induced_points(pts_nm, BCM_objects, lam_um):
    """Exterior induced potential at arbitrary points."""
    pts_nm = np.asarray(pts_nm, dtype=np.float64)
    Phi    = np.zeros(len(pts_nm), dtype=np.complex128)
    for obj in BCM_objects:
        R      = obj.diameter / 2.0
        X      = obj.coef_at(lam_um)
        li, mi = _lm_arrays(obj.lmax)
        if X.shape[0] != li.size:
            raise ValueError(f"{obj.label}: coef size mismatch")
        dx = pts_nm[:,0] - obj.position[0]
        dy = pts_nm[:,1] - obj.position[1]
        dz = pts_nm[:,2] - obj.position[2]
        r, th, ph = _cart2sph(dx, dy, dz)
        outside = r > R * 1.000001
        if not np.any(outside): continue
        r_o, th_o, ph_o = r[outside], th[outside], ph[outside]
        Y      = em_sph_harm(mi[:,None], li[:,None], th_o[None,:], ph_o[None,:])
        pref   = R**(li + 2) / eps0 / np.sqrt((2.0*li + 1.0) * R**3)
        radial = r_o[None,:]**(li[:,None] + 1)
        Phi[outside] += np.sum((pref * X)[:,None] * (Y / radial), axis=0)
    return Phi


def _efield_plane_grid(plane, origin_nm, span_nm, n):
    u = np.linspace(-span_nm, span_nm, n)
    v = np.linspace(-span_nm, span_nm, n)
    U, V  = np.meshgrid(u, v, indexing="xy")
    ox, oy, oz = origin_nm
    if plane == "yz":
        Xg = np.full_like(U, ox); Yg = U + oy; Zg = V + oz
        lab1, lab2 = "y (nm)", "z (nm)"
    elif plane == "xz":
        Xg = U + ox; Yg = np.full_like(U, oy); Zg = V + oz
        lab1, lab2 = "x (nm)", "z (nm)"
    elif plane == "xy":
        Xg = U + ox; Yg = V + oy; Zg = np.full_like(U, oz)
        lab1, lab2 = "x (nm)", "y (nm)"
    else:
        raise ValueError("plane must be 'yz', 'xz', or 'xy'")
    pts = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1)
    return U, V, pts, u[1]-u[0], v[1]-v[0], lab1, lab2, (Xg, Yg, Zg)


def host_mask_from_xyz(Xg, Yg, Zg, BCM_objects, shell_nm=0.0):
    mask = np.ones_like(Xg, dtype=bool)
    for obj in BCM_objects:
        R  = obj.diameter / 2.0
        dx = Xg - obj.position[0]; dy = Yg - obj.position[1]; dz = Zg - obj.position[2]
        mask &= (np.sqrt(dx*dx + dy*dy + dz*dz) > R + shell_nm)
    return mask


def _host_gradient_2d(Phi, mask, du, dv):
    Pp_u = np.roll(Phi, -1, axis=1); Pm_u = np.roll(Phi, 1, axis=1)
    Pp_v = np.roll(Phi, -1, axis=0); Pm_v = np.roll(Phi, 1, axis=0)
    mp_u = np.roll(mask, -1, axis=1); mm_u = np.roll(mask, 1, axis=1)
    mp_v = np.roll(mask, -1, axis=0); mm_v = np.roll(mask, 1, axis=0)
    dPhi_du = (Pp_u - Pm_u) / (2*du)
    dPhi_dv = (Pp_v - Pm_v) / (2*dv)
    dPhi_du = np.where(mp_u & ~mm_u, (Pp_u - Phi) / du, dPhi_du)
    dPhi_du = np.where(mm_u & ~mp_u, (Phi - Pm_u) / du, dPhi_du)
    dPhi_dv = np.where(mp_v & ~mm_v, (Pp_v - Phi) / dv, dPhi_dv)
    dPhi_dv = np.where(mm_v & ~mp_v, (Phi - Pm_v) / dv, dPhi_dv)
    dPhi_du[:, 0] = 0;  dPhi_du[:, -1] = 0
    dPhi_dv[0, :] = 0;  dPhi_dv[-1, :]  = 0
    nan = np.nan + 0j
    return np.where(mask, dPhi_du, nan), np.where(mask, dPhi_dv, nan)


_EFIELD_PLANE_NORMAL = {
    "yz": np.array([1., 0., 0.]),
    "xz": np.array([0., 1., 0.]),
    "xy": np.array([0., 0., 1.]),
}


def compute_E2_maps_host_only(
    *, BCM_objects, efield, lam_um,
    plane="yz", origin_nm=(0., 0., 0.),
    span_nm=25., n=301, shell_nm=0., delta_nm=None,
):
    """Compute |E|² maps in the host medium (exterior to all spheres)."""
    p = plane.lower(); nv = _EFIELD_PLANE_NORMAL[p]; o = np.asarray(origin_nm, float)
    U, V, pts, du, dv, lab1, lab2, (Xg, Yg, Zg) = _efield_plane_grid(p, o, span_nm, n)
    mask = host_mask_from_xyz(Xg, Yg, Zg, BCM_objects, shell_nm)
    if delta_nm is None: delta_nm = min(du, dv)

    Phi_0 = phi_induced_points(pts, BCM_objects, lam_um).reshape(n, n)
    _, _, pts_p, *_ = _efield_plane_grid(p, o + delta_nm*nv, span_nm, n)
    _, _, pts_m, *_ = _efield_plane_grid(p, o - delta_nm*nv, span_nm, n)
    Phi_p = phi_induced_points(pts_p, BCM_objects, lam_um).reshape(n, n)
    Phi_m = phi_induced_points(pts_m, BCM_objects, lam_um).reshape(n, n)

    dPhi_du, dPhi_dv = _host_gradient_2d(Phi_0, mask, du, dv)
    dPhi_dn = np.where(mask, (Phi_p - Phi_m) / (2*delta_nm), np.nan + 0j)

    Ex = np.zeros((n, n), dtype=np.complex128)
    Ey = np.zeros((n, n), dtype=np.complex128)
    Ez = np.zeros((n, n), dtype=np.complex128)
    if p == "yz":    Ey = -dPhi_du; Ez = -dPhi_dv; Ex = -dPhi_dn
    elif p == "xz":  Ex = -dPhi_du; Ez = -dPhi_dv; Ey = -dPhi_dn
    elif p == "xy":  Ex = -dPhi_du; Ey = -dPhi_dv; Ez = -dPhi_dn

    E0v = efield.E0 * np.asarray(efield.e_hat, float)
    Ex += E0v[0]; Ey += E0v[1]; Ez += E0v[2]

    E2      = np.where(mask, (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2).real, np.nan)
    E2_over = E2 / float(efield.E0**2)
    return U, V, E2, E2_over, (lab1, lab2)


def _make_sphere_img(n=200, base_rgb=(0.70, 0.70, 0.73)):
    """RGBA image of a Phong-shaded metallic sphere."""
    t    = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(t, t)
    r2   = X**2 + Y**2;  ins = r2 < 1.0
    with np.errstate(invalid='ignore'):
        Z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    lx, ly, lz = -0.40, 0.60, 0.69
    dot  = np.clip(lx*X + ly*Y + lz*Z, 0.0, 1.0)
    spec = np.where(ins, np.clip(2.0*dot*Z - lz, 0.0, 1.0)**14, 0.0)
    b    = np.clip(0.22 + 0.58*dot + 0.48*spec, 0.0, 1.0)
    img  = np.zeros((n, n, 4), dtype=np.float32)
    for c, bc in enumerate(base_rgb):
        img[:, :, c] = bc * b
    img[:, :, 3] = ins.astype(np.float32)
    return img

_SPHERE_IMG = _make_sphere_img()


def _draw_sphere_circles(ax, BCM_objects, plane, origin_nm=(0., 0., 0.)):
    p = plane.lower(); ox, oy, oz = origin_nm
    for obj in BCM_objects:
        R   = obj.diameter / 2.0;  pos = obj.position
        if   p == "yz":  cut_dist, cx, cy = pos[0]-ox, pos[1], pos[2]
        elif p == "xz":  cut_dist, cx, cy = pos[1]-oy, pos[0], pos[2]
        elif p == "xy":  cut_dist, cx, cy = pos[2]-oz, pos[0], pos[1]
        else: continue
        if abs(cut_dist) >= R: continue
        r_app = np.sqrt(R**2 - cut_dist**2)
        ax.imshow(_SPHERE_IMG,
                  extent=[cx-r_app, cx+r_app, cy-r_app, cy+r_app],
                  origin='lower', aspect='auto', zorder=10, interpolation='bilinear')
        ax.add_patch(Circle((cx, cy), r_app, fc='none', ec='#303030', lw=0.9, zorder=11))


def plot_two_maps(U, V, E2, E2o, lam_um, labels, title_prefix="",
                  BCM_objects=None, plane=None, origin_nm=(0., 0., 0.)):
    """Plot exterior |E|² and |E|²/|E0|² maps side by side."""
    lab1, lab2 = labels
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(alpha=0.0)

    for data, title, cblabel in [
        (E2,  rf"$|E|^2$ at $\lambda$={lam_um*1e3:.1f} nm",         r"$|E|^2$ (V/nm)²"),
        (E2o, rf"$|E|^2/|E_0|^2$ at $\lambda$={lam_um*1e3:.1f} nm", r"$|E|^2/|E_0|^2$"),
    ]:
        vmin = np.nanpercentile(data, 0.5)
        vmax = np.nanpercentile(data, 99.5)
        vmin = min(vmin, 0.999)
        vmax = max(vmax, 1.001)
        norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor('#383838')
        pcm = ax.pcolormesh(U, V, data, shading="auto", cmap=cmap, norm=norm, zorder=1)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if BCM_objects is not None and plane is not None:
            _draw_sphere_circles(ax, BCM_objects, plane, origin_nm)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

        ax.set_xlabel(lab1); ax.set_ylabel(lab2)
        ax.set_title(title_prefix + title)
        fig.colorbar(pcm, ax=ax, label=cblabel)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        plt.show()


def phi_all_points(pts_nm, BCM_objects, lam_um):
    """Induced potential at arbitrary points (interior + exterior)."""
    pts = np.asarray(pts_nm, float)
    Phi = np.zeros(len(pts), dtype=np.complex128)

    owner = -np.ones(len(pts), dtype=int)
    for j, obj in enumerate(BCM_objects):
        R  = obj.diameter / 2.0
        dx = pts[:,0] - obj.position[0]
        dy = pts[:,1] - obj.position[1]
        dz = pts[:,2] - obj.position[2]
        owner[np.sqrt(dx*dx + dy*dy + dz*dz) < R * 0.999999] = j

    for j, obj in enumerate(BCM_objects):
        R      = obj.diameter / 2.0
        X      = obj.coef_at(lam_um)
        li, mi = _lm_arrays(obj.lmax)

        dx = pts[:,0] - obj.position[0]
        dy = pts[:,1] - obj.position[1]
        dz = pts[:,2] - obj.position[2]
        r, th, ph = _cart2sph(dx, dy, dz)

        ins = (owner == j)
        if np.any(ins):
            r_in = np.maximum(r[ins], 1e-30)
            Y    = em_sph_harm(mi[:,None], li[:,None],
                               th[ins][None,:], ph[ins][None,:])
            rho  = r_in[None,:] / R
            pref = 1.0 / (eps0 * np.sqrt((2.0*li+1.0) * R))
            Phi[ins] += np.sum((pref * X)[:,None] * Y * rho**li[:,None], axis=0)

        out = ~ins & (r > R * 1.000001)
        if np.any(out):
            r_out = r[out]
            Y     = em_sph_harm(mi[:,None], li[:,None],
                                th[out][None,:], ph[out][None,:])
            pref  = R**(li+2) / eps0 / np.sqrt((2.0*li+1.0) * R**3)
            Phi[out] += np.sum((pref * X)[:,None] * Y / r_out[None,:]**(li[:,None]+1), axis=0)

    return Phi


def compute_E2_maps_interior(
    *, BCM_objects, efield, lam_um,
    plane="yz", origin_nm=(0., 0., 0.),
    span_nm=25., n=301, delta_nm=None,
):
    """Compute |E|² maps inside the metal spheres."""
    p = plane.lower(); nv = _EFIELD_PLANE_NORMAL[p]; o = np.asarray(origin_nm, float)
    U, V, pts, du, dv, lab1, lab2, (Xg, Yg, Zg) = _efield_plane_grid(p, o, span_nm, n)

    metal_mask = ~host_mask_from_xyz(Xg, Yg, Zg, BCM_objects, shell_nm=0.0)

    if delta_nm is None:
        delta_nm = min(du, dv)

    # shrink by one grid step so in-plane finite differences never cross the
    # metal-vacuum interface (where the potential has a kink)
    shell_nm   = max(du, dv)
    grad_mask  = ~host_mask_from_xyz(Xg, Yg, Zg, BCM_objects, shell_nm=-shell_nm)

    Phi_0 = phi_all_points(pts, BCM_objects, lam_um).reshape(n, n)
    _, _, pts_p, *_ = _efield_plane_grid(p, o + delta_nm*nv, span_nm, n)
    _, _, pts_m, *_ = _efield_plane_grid(p, o - delta_nm*nv, span_nm, n)
    Phi_p = phi_all_points(pts_p, BCM_objects, lam_um).reshape(n, n)
    Phi_m = phi_all_points(pts_m, BCM_objects, lam_um).reshape(n, n)

    dPhi_du, dPhi_dv = _host_gradient_2d(Phi_0, grad_mask, du, dv)
    dPhi_dn = np.where(metal_mask, (Phi_p - Phi_m) / (2*delta_nm), np.nan + 0j)

    Ex = np.zeros((n, n), dtype=np.complex128)
    Ey = np.zeros((n, n), dtype=np.complex128)
    Ez = np.zeros((n, n), dtype=np.complex128)
    if p == "yz":    Ey = -dPhi_du; Ez = -dPhi_dv; Ex = -dPhi_dn
    elif p == "xz":  Ex = -dPhi_du; Ez = -dPhi_dv; Ey = -dPhi_dn
    elif p == "xy":  Ex = -dPhi_du; Ey = -dPhi_dv; Ez = -dPhi_dn

    E0v = efield.E0 * np.asarray(efield.e_hat, float)
    Ex += E0v[0]; Ey += E0v[1]; Ez += E0v[2]

    E2      = np.where(metal_mask, (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2).real, np.nan)
    E2_over = E2 / float(efield.E0**2)
    return U, V, E2, E2_over, (lab1, lab2)


def plot_interior_maps(U, V, E2, E2o, lam_um, labels,
                       BCM_objects=None, plane=None, origin_nm=(0., 0., 0.)):
    """Plot interior |E|² maps."""
    lab1, lab2 = labels
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(alpha=0.0)

    for data, title, cblabel in [
        (E2,  rf"$|E_{{\mathrm{{int}}}}|^2$ at $\lambda$={lam_um*1e3:.1f} nm",         r"$|E|^2$ (V/nm)²"),
        (E2o, rf"$|E_{{\mathrm{{int}}}}|^2/|E_0|^2$ at $\lambda$={lam_um*1e3:.1f} nm", r"$|E|^2/|E_0|^2$"),
    ]:
        vmin = np.nanpercentile(data, 0.5)
        vmax = np.nanpercentile(data, 99.5)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor('white')
        pcm  = ax.pcolormesh(U, V, data, shading="auto", cmap=cmap,
                              vmin=vmin, vmax=vmax, zorder=1)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if BCM_objects is not None and plane is not None:
            p = plane.lower(); ox, oy, oz = origin_nm
            for obj in BCM_objects:
                R = obj.diameter / 2.0; pos = obj.position
                if   p == "yz":  cut_dist, cx, cy = pos[0]-ox, pos[1], pos[2]
                elif p == "xz":  cut_dist, cx, cy = pos[1]-oy, pos[0], pos[2]
                elif p == "xy":  cut_dist, cx, cy = pos[2]-oz, pos[0], pos[1]
                else: continue
                if abs(cut_dist) >= R: continue
                r_app = np.sqrt(R**2 - cut_dist**2)
                ax.add_patch(Circle((cx, cy), r_app,
                                    fc='none', ec='#303030', lw=1.0, zorder=10))
        ax.set_xlim(xlim); ax.set_ylim(ylim)

        ax.set_xlabel(lab1); ax.set_ylabel(lab2)
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label=cblabel)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        plt.show()


def _best_plane(e_hat, k_hat=None):
    e = np.asarray(e_hat, float); e /= np.linalg.norm(e)
    scores = {pl: abs(float(np.dot(nv, e))) for pl, nv in _EFIELD_PLANE_NORMAL.items()}
    best_score = min(scores.values())
    candidates = [pl for pl, s in scores.items() if np.isclose(s, best_score, atol=1e-6)]
    if len(candidates) == 1 or k_hat is None: return candidates[0]
    k = np.asarray(k_hat, float); k /= np.linalg.norm(k)
    return min(candidates, key=lambda pl: abs(float(np.dot(_EFIELD_PLANE_NORMAL[pl], k))))


def _cluster_span(BCM_objects, margin_nm=3.0):
    r_max = max(np.linalg.norm(obj.position) + obj.diameter / 2.0
                for obj in BCM_objects)
    return r_max + margin_nm


def plot_efield_maps(results, *, plane=None, origin_nm=(0, 0, 0),
                     margin_nm=4.0, n=101, show=True):
    """
    Compute and plot exterior + interior E-field maps at each absorption peak.

    Parameters
    ----------
    results : dict
        Output of ``bcm.run_optical_response(config)``.
    plane : str or None
        Cut plane ('yz', 'xz', 'xy'). If None, auto-selected from efield.
    origin_nm : tuple
        Origin of the cut plane.
    margin_nm : float
        Extra margin beyond the cluster span.
    n : int
        Grid resolution per axis.
    show : bool
        Whether to call plt.show() (handled inside plot_two_maps / plot_interior_maps).
    """
    BCM_objects    = results['BCM_objects']
    efield         = results['efield']
    lam_um         = results['lam_um']
    peak_idx_total = results['peak_idx_total']

    if plane is None:
        plane = _best_plane(efield.e_hat, efield.k_hat)

    span = _cluster_span(BCM_objects, margin_nm=margin_nm)

    print(f"Using {len(peak_idx_total)} absorption peaks from optical response")
    print(f"plane = {plane}  |  span = {span:.1f} nm\n")

    for i, idx in enumerate(peak_idx_total):
        lam0 = float(lam_um[idx])
        print(f"── Peak {i+1}/{len(peak_idx_total)}: λ = {lam0*1e3:.1f} nm ──")

        U, V, E2, E2o, labels = compute_E2_maps_host_only(
            BCM_objects=BCM_objects, efield=efield, lam_um=lam0,
            plane=plane, origin_nm=origin_nm, span_nm=span, n=n,
        )
        plot_two_maps(U, V, E2, E2o, lam0, labels,
                      BCM_objects=BCM_objects, plane=plane, origin_nm=origin_nm)

        U_i, V_i, E2_i, E2o_i, lab_i = compute_E2_maps_interior(
            BCM_objects=BCM_objects, efield=efield, lam_um=lam0,
            plane=plane, origin_nm=origin_nm, span_nm=span, n=n,
        )
        plot_interior_maps(U_i, V_i, E2_i, E2o_i, lam0, lab_i,
                           BCM_objects=BCM_objects, plane=plane, origin_nm=origin_nm)




# === INTERACTIVE BUILDER (moved from interactive_builder.py) ===
"""
Interactive nanocluster geometry builder with real-time 3D visualization.

This module provides an interactive widget-based interface for designing
nanocluster geometries with live 3D visualization using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from ipywidgets import (
    VBox, HBox, Button, Output,
    IntSlider, FloatSlider, Dropdown, Label, Tab, FloatText, Checkbox
)
from IPython.display import display, clear_output
from pathlib import Path

import plytrons.bcm_sphere as bcm
from plytrons.bcm_sphere import EField, BCMObject



# =============================================================================
# MATERIAL LIBRARY
# =============================================================================
MATERIALS = {
    'Silver': {
        'EF': 5.53,      # Fermi energy (eV)
        'wp': 9.07,      # Plasma frequency (eV)
        'vf': 1.39e6,    # Fermi velocity (m/s)
        'gamma0': 0.060, # Damping
        'eps_b': 4.18,   # Background permittivity
        'color': '#C0C0C0'
    },
    'Gold': {
        'EF': 5.53,
        'wp': 9.03,
        'vf': 1.40e6,
        'gamma0': 0.070,
        'eps_b': 9.84,
        'color': '#FFD700'
    },
    'Copper': {
        'EF': 7.0,
        'wp': 8.8,
        'vf': 1.57e6,
        'gamma0': 0.027,
        'eps_b': 1.0,
        'color': '#B87333'
    }
}

# Refractive-index presets for the surrounding medium
MEDIA = {
    'Vacuum / Air': 1.0,
    'Water':        1.77,   # n=1.33  → eps = n²
    'Glass (SiO₂)': 2.25,  # n=1.50
    'TiO₂':         6.25,  # n=2.50
}


# =============================================================================
# GEOMETRY PRESETS
# =============================================================================
def get_preset_geometry(preset_name, D, delta):
    """
    Generate preset nanocluster geometries.

    Parameters
    ----------
    preset_name : str
        Name of preset geometry
    D : float
        Diameter of spheres (nm)
    delta : float
        Gap between sphere surfaces (nm)

    Returns
    -------
    list of np.ndarray
        List of 3D positions for each particle
    """
    d_c = D + delta

    if preset_name == 'Monomer':
        return [np.array([0, 0, 0])]

    elif preset_name == 'Dimer':
        return [
            np.array([0, d_c/2, 0]),
            np.array([0, -d_c/2, 0])
        ]

    elif preset_name == 'Trimer':
        return [
            np.array([0, d_c/2, 0]),
            np.array([0, -d_c/2, 0]),
            np.array([0, 0, -(d_c/2)*np.sqrt(3)])
        ]

    elif preset_name in ('Tetramer', 'Tetramer (square)'):
        # Square planar: 4 particles at corners of a square with edge d_c
        return [
            np.array([0,  d_c/2,  d_c/2]),
            np.array([0,  d_c/2, -d_c/2]),
            np.array([0, -d_c/2,  d_c/2]),
            np.array([0, -d_c/2, -d_c/2]),
        ]

    elif preset_name == 'Tetramer (tetrahedron)':
        # Regular tetrahedron: all 6 edges = d_c
        # Vertices: (±1,±1,±1) with even permutations, scaled so edge = d_c
        s = d_c / (2 * np.sqrt(2))
        return [
            np.array([ s,  s,  s]),
            np.array([ s, -s, -s]),
            np.array([-s,  s, -s]),
            np.array([-s, -s,  s]),
        ]

    elif preset_name in ('Pentamer (planar)', 'Pentamer (ring)'):
        N = 5
        return [
            np.array([
                0,
                (d_c/(2*np.sin(np.pi/N)))*np.cos(np.pi/2 + 2*np.pi/N*i),
                (d_c/(2*np.sin(np.pi/N)))*np.sin(np.pi/2 + 2*np.pi/N*i)
            ]) for i in range(N)
        ]

    elif preset_name == 'Pentamer (pyramid)':
        # Square base (edge d_c) + apex, all lateral edges = d_c
        h = d_c / np.sqrt(2)   # apex height above base so lateral edge = d_c
        return [
            np.array([0,  d_c/2,  d_c/2]),
            np.array([0,  d_c/2, -d_c/2]),
            np.array([0, -d_c/2,  d_c/2]),
            np.array([0, -d_c/2, -d_c/2]),
            np.array([h,  0,      0     ]),   # apex along x
        ]

    elif preset_name == 'Hexamer (ring)':
        N = 6
        return [
            np.array([
                0,
                (d_c/(2*np.sin(np.pi/N)))*np.cos(np.pi/2 + 2*np.pi/N*i),
                (d_c/(2*np.sin(np.pi/N)))*np.sin(np.pi/2 + 2*np.pi/N*i)
            ]) for i in range(N)
        ]

    elif preset_name == 'Hexamer (bi-triangle)':
        # Two equilateral triangles (edge d_c) stacked and rotated 60° — triangular antiprism
        R = d_c / np.sqrt(3)    # circumradius of equilateral triangle with edge d_c
        h = d_c / 2             # half-height between layers
        bottom = [
            np.array([0, R * np.cos(np.pi/2 + 2*np.pi/3*i),
                         R * np.sin(np.pi/2 + 2*np.pi/3*i), ]) for i in range(3)
        ]
        top = [
            np.array([0, R * np.cos(np.pi/2 + np.pi/3 + 2*np.pi/3*i),
                         R * np.sin(np.pi/2 + np.pi/3 + 2*np.pi/3*i)]) for i in range(3)
        ]
        # Reorder as [x, y, z] with x being the stacking axis
        bottom = [np.array([-h, p[1], p[2]]) for p in bottom]
        top    = [np.array([ h, p[1], p[2]]) for p in top]
        return bottom + top

    elif preset_name == 'Heptamer (ring + center)':
        N = 6
        positions = [np.array([0, 0, 0])]  # center particle
        positions.extend([
            np.array([
                0,
                (d_c/(2*np.sin(np.pi/N)))*np.cos(np.pi/2 + 2*np.pi/N*i),
                (d_c/(2*np.sin(np.pi/N)))*np.sin(np.pi/2 + 2*np.pi/N*i)
            ]) for i in range(N)
        ])
        return positions

    elif preset_name == 'Octahedron':
        # Regular octahedron: all 12 edges = d_c
        # 6 vertices along ±x, ±y, ±z at distance d_c/√2 from origin
        a = d_c / np.sqrt(2)
        return [
            np.array([ a,  0,  0]),
            np.array([-a,  0,  0]),
            np.array([ 0,  a,  0]),
            np.array([ 0, -a,  0]),
            np.array([ 0,  0,  a]),
            np.array([ 0,  0, -a]),
        ]

    elif preset_name == 'Satellite (1+6)':
        # 1 central particle + 6 shell particles along ±x, ±y, ±z at distance d_c
        return [
            np.array([0,    0,    0   ]),   # core
            np.array([d_c,  0,    0   ]),
            np.array([-d_c, 0,    0   ]),
            np.array([0,    d_c,  0   ]),
            np.array([0,   -d_c,  0   ]),
            np.array([0,    0,    d_c ]),
            np.array([0,    0,   -d_c ]),
        ]

    return [np.array([0, 0, 0])]


# =============================================================================
# CLUSTER BUILDER  (returns BCMObject instances)
# =============================================================================
def build_cluster(structure, D, delta, lmax, eps_func, eps_h=1.0):
    """
    Build a list of BCMObject instances for standard nanocluster geometries.

    All spheres are identical (diameter D, lmax, eps_func).  Every
    nearest-neighbor centre-to-centre distance equals D + delta exactly.
    All clusters lie in the **yz-plane** (x = 0 for every sphere).

    Parameters
    ----------
    structure : int
        Number of spheres.  Supported values:

        * **2** — dimer, axis along y
        * **3** — equilateral trimer in the yz-plane
        * **4** — square planar in the yz-plane (edge = D + delta)
        * **5** — regular pentagon in the yz-plane
        * **7** — heptamer: 6-sphere hexagonal ring + 1 central sphere

    D : float
        Sphere diameter [nm].
    delta : float
        Surface-to-surface gap between nearest neighbours [nm].
        Must be > 0.
    lmax : int
        Maximum spherical-harmonic order for each BCMObject.
    eps_func : callable
        Permittivity function ``eps(lam_um) -> complex``.
    eps_h : float, optional
        Host-medium permittivity (default 1.0, vacuum).

    Returns
    -------
    list of BCMObject
        Ready-to-use objects with positions verified for correct NN distance.

    Raises
    ------
    ValueError
        If ``structure`` is not in ``{2, 3, 4, 5, 7}``.
    ValueError
        If ``delta`` is not positive.

    Notes
    -----
    Geometry derivations (d_c = D + delta):

    * **Dimer (2):**  positions ``(0, ±d_c/2, 0)``.
    * **Trimer (3):**  equilateral triangle with edge d_c;
      circumradius ``R = d_c / √3``; vertices at
      ``(0, R cos(π/2 + 2πk/3), R sin(π/2 + 2πk/3))`` for k = 0, 1, 2.
    * **Tetramer (4):**  square with edge d_c;
      vertices at ``(0, ±d_c/2, ±d_c/2)``.
    * **Pentamer (5):**  regular pentagon with edge d_c;
      circumradius ``R = d_c / (2 sin(π/5))``.
    * **Heptamer (7):**  centre at origin + hexagonal ring with edge d_c;
      circumradius ``R = d_c / (2 sin(π/6)) = d_c``; ring vertices at
      ``(0, d_c cos(π/2 + 2πk/6), d_c sin(π/2 + 2πk/6))`` for k = 0…5.
    """
    _SUPPORTED = {2, 3, 4, 5, 7}
    if structure not in _SUPPORTED:
        raise ValueError(
            f"structure={structure} is not supported. "
            f"Choose from {sorted(_SUPPORTED)}."
        )
    if delta <= 0:
        raise ValueError(f"delta must be positive; got {delta}.")

    d_c = float(D) + float(delta)   # nearest-neighbour centre-to-centre distance

    # ── Position tables ───────────────────────────────────────────────────
    if structure == 2:
        positions = [
            np.array([0.0,  d_c / 2, 0.0]),
            np.array([0.0, -d_c / 2, 0.0]),
        ]

    elif structure == 3:
        # Equilateral triangle, edge = d_c.
        # Circumradius: R = d_c / sqrt(3)
        R = d_c / np.sqrt(3)
        positions = [
            np.array([0.0,
                      R * np.cos(np.pi / 2 + 2 * np.pi / 3 * k),
                      R * np.sin(np.pi / 2 + 2 * np.pi / 3 * k)])
            for k in range(3)
        ]

    elif structure == 4:
        # Square, edge = d_c.
        h = d_c / 2
        positions = [
            np.array([0.0,  h,  h]),
            np.array([0.0,  h, -h]),
            np.array([0.0, -h,  h]),
            np.array([0.0, -h, -h]),
        ]

    elif structure == 5:
        # Regular pentagon, edge = d_c.
        # Circumradius: R = d_c / (2 sin(π/5))
        R = d_c / (2 * np.sin(np.pi / 5))
        positions = [
            np.array([0.0,
                      R * np.cos(np.pi / 2 + 2 * np.pi / 5 * k),
                      R * np.sin(np.pi / 2 + 2 * np.pi / 5 * k)])
            for k in range(5)
        ]

    else:  # structure == 7
        # Heptamer: centre + hexagonal ring, edge = d_c.
        # Circumradius of regular hexagon with edge d_c: R = d_c
        R = d_c
        ring = [
            np.array([0.0,
                      R * np.cos(np.pi / 2 + 2 * np.pi / 6 * k),
                      R * np.sin(np.pi / 2 + 2 * np.pi / 6 * k)])
            for k in range(6)
        ]
        positions = [np.array([0.0, 0.0, 0.0])] + ring

    # ── Sanity check: every NN pair has distance d_c ──────────────────────
    # Build NN table: for each sphere find the minimum distance to any other.
    pos_arr = np.array(positions)
    for i, pi in enumerate(pos_arr):
        dists = [np.linalg.norm(pi - pos_arr[j]) for j in range(len(positions)) if j != i]
        nn_dist = min(dists)
        if abs(nn_dist - d_c) > 1e-9:
            raise RuntimeError(
                f"build_cluster: internal geometry error for structure={structure}. "
                f"Sphere {i} nearest-neighbour distance = {nn_dist:.6g} nm, "
                f"expected {d_c:.6g} nm."
            )

    # ── Assemble BCMObject list ────────────────────────────────────────────
    from plytrons.bcm_sphere import BCMObject as _BCMObject
    return [
        _BCMObject(
            label=f'Sphere{i + 1}',
            diameter=float(D),
            lmax=int(lmax),
            eps=eps_func,
            position=pos,
        )
        for i, pos in enumerate(positions)
    ]


# =============================================================================
# POLARIZATION PRESETS
# =============================================================================
def get_polarizations(structure):
    """
    Return the physically distinct (k_hat, e_hat) illumination tuples for
    a given cluster geometry.

    "Distinct" means that no two entries are related by a symmetry operation
    of the cluster that maps the cluster onto itself — so running the full
    set covers all inequivalent optical responses without redundancy.

    Parameters
    ----------
    structure : int
        Number of spheres.  Must be in {2, 3, 4, 5, 7}.

    Returns
    -------
    list of tuple (k_hat, e_hat)
        Each entry is a pair of unit-vector arrays (shape (3,), float64).
        Both vectors are already normalised.

    Raises
    ------
    ValueError
        If ``structure`` is not in {2, 3, 4, 5, 7}.

    Symmetry notes
    --------------
    All clusters lie in the **yz-plane** (x = 0).  The illumination
    direction k and polarisation E are always perpendicular.

    **Dimer (2) — C_{2v}, axis along y**
        Two inequivalent channels:
        * Longitudinal (E ∥ y): drives the bonding (bright) plasmon mode;
          near-field concentrates in the gap.
        * Transverse   (E ∥ z): drives the antibonding / transverse mode;
          weak gap coupling.

    **Equilateral trimer (3) — D_{3h}, yz-plane**
        Three distinct channels:
        * In-plane longitudinal (k ∥ x, E ∥ y): E lies along one
          symmetry axis of the triangle.
        * In-plane transverse   (k ∥ x, E ∥ z): E lies perpendicular
          to that axis but still in-plane; inequivalent because the
          triangle is not square.
        * Out-of-plane          (k ∥ y, E ∥ x): E is normal to the
          cluster plane; drives out-of-plane (dark/toroidal) modes.

    **Square tetramer (4) — D_{4h}, yz-plane**
        Same three distinct channels as the trimer.  The 4-fold in-plane
        symmetry makes the two orthogonal in-plane directions (E ∥ y and
        E ∥ z, both with k ∥ x) equivalent by a 90° rotation, **but** for
        a square we keep both because the square has only C_4, so E ∥ face-
        diagonal vs E ∥ edge are distinct.  Out-of-plane (E ∥ x) is the
        third independent channel.

    **Regular pentagon (5) — D_{5h}, yz-plane**
        Same three channels as the trimer.  The 5-fold symmetry makes all
        in-plane E directions related *within* each k direction, so k ∥ x
        with E ∥ y and k ∥ x with E ∥ z are the only two in-plane
        inequivalent choices; out-of-plane k ∥ y, E ∥ x is the third.

    **Heptamer (7) — C_{6v}, yz-plane**
        Two distinct channels:
        * In-plane    (k ∥ x, E ∥ y): the 6-fold ring symmetry makes
          all in-plane E orientations equivalent up to a rotation of the
          cluster — a single representative suffices.
        * Out-of-plane (k ∥ x, E ∥ z): E normal to the ring plane;
          drives the dark out-of-plane mode.
          Note: k ∥ x with E ∥ z is the same channel as k ∥ y with
          E ∥ x after a 90° rotation about z, so only one entry is kept.
    """
    _SUPPORTED = {2, 3, 4, 5, 7}
    if structure not in _SUPPORTED:
        raise ValueError(
            f"structure={structure} is not supported. "
            f"Choose from {sorted(_SUPPORTED)}."
        )

    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])

    if structure == 2:
        # Dimer axis along y: longitudinal (E∥y) and transverse (E∥z)
        return [
            (x.copy(), y.copy()),   # longitudinal
            (x.copy(), z.copy()),   # transverse
        ]

    elif structure in (3, 4, 5):
        # Trimer / square tetramer / pentagon: all lie in yz-plane.
        # Three inequivalent channels: two in-plane, one out-of-plane.
        return [
            (x.copy(), y.copy()),   # in-plane, E along y
            (x.copy(), z.copy()),   # in-plane, E along z
            (y.copy(), x.copy()),   # out-of-plane, E along x
        ]

    else:  # structure == 7
        # Heptamer: 6-fold symmetry collapses all in-plane E directions;
        # one in-plane and one out-of-plane channel suffice.
        return [
            (x.copy(), y.copy()),   # in-plane
            (x.copy(), z.copy()),   # out-of-plane (E∥z normal to ring plane)
        ]


# =============================================================================
# 3D VISUALIZATION HELPERS
# =============================================================================
def create_sphere_mesh(center, radius, resolution=20):
    """Create mesh coordinates for a sphere."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_arrow(start, direction, length, color, name):
    """Create 3D arrow with shaft and cone head."""
    end = start + direction * length

    # Arrow shaft
    shaft = go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=8),
        name=name,
        showlegend=True,
        hoverinfo='name'
    )

    # Arrow head (cone)
    cone = go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=length*0.2,
        name=name,
        showlegend=False,
        hoverinfo='skip'
    )

    return [shaft, cone]


# =============================================================================
# INTERACTIVE BUILDER CLASS
# =============================================================================
class NanoclusterBuilder:
    """
    Interactive widget for designing nanocluster geometries.

    Features:
    - Real-time 3D visualization (inline or separate window)
    - Preset geometries (monomer through heptamer)
    - Individual particle control (position, size, material)
    - Field vector visualization (k, E, H)
    - Direct export to BCM objects

    Parameters
    ----------
    display_mode : str, optional
        How to display plots: 'inline' (in notebook), 'browser' (separate window),
        or 'auto' (try browser, fall back to inline). Default: 'inline'

    Examples
    --------
    >>> builder = NanoclusterBuilder(display_mode='inline')
    >>> # Adjust parameters in the GUI
    >>> BCM_objects = builder.export_bcm_objects(eps_drude, lmax=10)
    >>> efield = builder.export_efield()
    """

    def __init__(self, display_mode='inline'):
        self.display_mode = display_mode  # 'inline', 'browser', or 'auto'
        self.output = Output()
        self.particle_widgets = []
        self.current_positions = []
        self.current_diameters = []
        self.current_materials = []
        self._updating_from_preset = False  # Flag to prevent preset->custom switching

        # Global parameters
        self.preset_dropdown = Dropdown(
            options=[
                'Custom',
                # ── Planar / 2-D ──────────────────────────────
                'Monomer',
                'Dimer',
                'Trimer',
                'Tetramer (square)',
                'Pentamer (ring)',
                'Hexamer (ring)',
                'Heptamer (ring + center)',
                # ── 3-D ───────────────────────────────────────
                'Tetramer (tetrahedron)',
                'Pentamer (pyramid)',
                'Hexamer (bi-triangle)',
                'Octahedron',
                'Satellite (1+6)',
            ],
            value='Dimer',
            description='Preset:',
            style={'description_width': '100px'}
        )

        self.n_particles = IntSlider(
            value=2, min=1, max=10, step=1,
            description='# Particles:',
            style={'description_width': '100px'}
        )

        self.default_diameter = FloatSlider(
            value=5.0, min=1.0, max=50.0, step=0.5,
            description='Default D (nm):',
            style={'description_width': '100px'}
        )

        self.gap = FloatSlider(
            value=0.5, min=0.1, max=5.0, step=0.1,
            description='Gap (nm):',
            style={'description_width': '100px'}
        )

        self.default_material = Dropdown(
            options=list(MATERIALS.keys()),
            value='Silver',
            description='Default Material:',
            style={'description_width': '100px'}
        )

        self.medium_dropdown = Dropdown(
            options=list(MEDIA.keys()),
            value='Vacuum / Air',
            description='Medium (εₕ):',
            style={'description_width': '100px'}
        )

        self.model_dropdown = Dropdown(
            options=['PWA', 'Nordlander', 'Bulk'],
            value='PWA',
            description='ε model:',
            style={'description_width': '100px'}
        )

        # Rendering quality
        self.sphere_resolution = IntSlider(
            value=25, min=10, max=50, step=5,
            description='Quality:',
            style={'description_width': '100px'},
            tooltip='Higher = better looking but slower'
        )

        # Field parameters - k-vector (propagation direction)
        self.k_x = FloatSlider(value=1, min=-1, max=1, step=0.1, description='k_x:',
                               style={'description_width': '50px'})
        self.k_y = FloatSlider(value=0, min=-1, max=1, step=0.1, description='k_y:',
                               style={'description_width': '50px'})
        self.k_z = FloatSlider(value=0, min=-1, max=1, step=0.1, description='k_z:',
                               style={'description_width': '50px'})

        # Polarization angle (rotation of E-vector in plane perpendicular to k)
        self.pol_angle = FloatSlider(
            value=90, min=0, max=360, step=5,
            description='Pol. angle (°):',
            style={'description_width': '100px'},
            tooltip='Rotation angle of E-vector in plane perpendicular to k'
        )

        # Display options
        self.show_field = Checkbox(value=True, description='Show E/H field')
        self.show_centers = Checkbox(value=True, description='Show centers')

        # Update button
        self.update_btn = Button(
            description='🔄 Update Geometry',
            button_style='primary',
            tooltip='Click to update 3D visualization'
        )
        self.update_btn.on_click(self.on_update_click)

        # Observers for auto-update on preset changes
        self.preset_dropdown.observe(self.on_preset_change, names='value')
        self.n_particles.observe(self.on_n_particles_change, names='value')
        self.gap.observe(self.on_gap_change, names='value')
        self.default_diameter.observe(self.on_diameter_change, names='value')
        self.default_material.observe(self.on_material_change, names='value')

        # Initialize UI
        self.setup_ui()
        self.update_plot()

    def on_preset_change(self, change):
        """Handle preset geometry selection."""
        if change['new'] != 'Custom':
            self._updating_from_preset = True  # Set flag
            D = self.default_diameter.value
            delta = self.gap.value
            positions = get_preset_geometry(change['new'], D, delta)
            self.n_particles.value = len(positions)
            self.current_positions = positions
            # Update diameters to match default
            self.current_diameters = [D] * len(positions)
            # Update materials to default
            self.current_materials = [self.default_material.value] * len(positions)
            self.update_particle_widgets()
            self.update_plot()
            self._updating_from_preset = False  # Clear flag

    def on_n_particles_change(self, change):
        """Handle change in number of particles."""
        # Only switch to Custom if not triggered by preset selection
        if not self._updating_from_preset:
            self.preset_dropdown.value = 'Custom'
        self.setup_particle_widgets()

    def on_gap_change(self, change):
        """Handle gap change for preset geometries."""
        if self.preset_dropdown.value != 'Custom':
            self.on_preset_change({'new': self.preset_dropdown.value})

    def on_diameter_change(self, change):
        """Handle default diameter change - update all particles."""
        new_diameter = change['new']
        # Update all current diameters
        self.current_diameters = [new_diameter] * len(self.current_positions)
        # Rebuild widgets with new diameters
        self.setup_particle_widgets()

    def on_material_change(self, change):
        """Handle default material change - update all particles."""
        new_material = change['new']
        # Update all current materials
        self.current_materials = [new_material] * len(self.current_positions)
        # Rebuild widgets with new materials and refresh the tab
        self.setup_particle_widgets()
        self.refresh_particles_tab()

    def setup_particle_widgets(self):
        """Create widgets for individual particle parameters."""
        n = self.n_particles.value

        # Initialize with current or default values
        if len(self.current_positions) != n:
            self.current_positions = [np.array([0.0, i*6.0, 0.0]) for i in range(n)]
            self.current_diameters = [self.default_diameter.value] * n
            self.current_materials = [self.default_material.value] * n

        self.particle_widgets = []

        for i in range(n):
            pos = self.current_positions[i] if i < len(self.current_positions) else np.array([0.0, i*6.0, 0.0])
            diam = self.current_diameters[i] if i < len(self.current_diameters) else self.default_diameter.value
            mat = self.current_materials[i] if i < len(self.current_materials) else self.default_material.value

            particle_box = VBox([
                Label(f'━━ Particle {i+1} ━━', layout={'width': '150px'}),
                FloatText(value=pos[0], description='x (nm):', step=0.1,
                         style={'description_width': '60px'}),
                FloatText(value=pos[1], description='y (nm):', step=0.1,
                         style={'description_width': '60px'}),
                FloatText(value=pos[2], description='z (nm):', step=0.1,
                         style={'description_width': '60px'}),
                FloatText(value=diam, description='D (nm):', step=0.5,
                         style={'description_width': '60px'}),
                Dropdown(options=list(MATERIALS.keys()), value=mat,
                        description='Material:',
                        style={'description_width': '60px'})
            ])
            self.particle_widgets.append(particle_box)

    def update_particle_widgets(self):
        """Update particle widget values from current positions."""
        self.setup_particle_widgets()
        # Refresh the particles tab
        self.refresh_particles_tab()

    def refresh_particles_tab(self):
        """Refresh the particles tab with updated widgets."""
        if hasattr(self, 'main_tab'):
            # Rebuild particles tab
            particles_tab = VBox([
                HBox(self.particle_widgets[:5]) if len(self.particle_widgets) > 1 else VBox(self.particle_widgets),
                HBox(self.particle_widgets[5:]) if len(self.particle_widgets) > 5 else VBox([])
            ])

            # Update the tab (keep other tabs intact)
            children = list(self.main_tab.children)
            children[1] = particles_tab  # Replace particles tab (index 1)
            self.main_tab.children = children

    def setup_ui(self):
        """Setup the complete user interface."""
        self.setup_particle_widgets()

        # Particles tab with horizontal layout for up to 5 particles
        particles_tab = VBox([
            HBox(self.particle_widgets[:5]) if len(self.particle_widgets) > 1 else VBox(self.particle_widgets),
            HBox(self.particle_widgets[5:]) if len(self.particle_widgets) > 5 else VBox([])
        ])

        # Global parameters tab
        global_tab = VBox([
            Label('━━━━━ Geometry Presets ━━━━━'),
            self.preset_dropdown,
            self.n_particles,
            self.default_diameter,
            self.gap,
            self.default_material,
            Label('━━━━━ Surrounding Medium ━━━━━'),
            self.medium_dropdown,
            Label('━━━━━ Dielectric Model ━━━━━'),
            self.model_dropdown,
            Label('━━━━━ Rendering ━━━━━'),
            self.sphere_resolution
        ])

        # Field and visualization tab
        field_tab = VBox([
            Label('━━━━━ Incident Wavevector k (propagation) ━━━━━'),
            HBox([self.k_x, self.k_y, self.k_z]),
            Label('━━━━━ Polarization (E-vector rotation) ━━━━━'),
            self.pol_angle,
            Label('   E-vector is computed perpendicular to k'),
            Label('   Angle rotates E in plane ⊥ k'),
            Label('━━━━━ Display Options ━━━━━'),
            self.show_field,
            self.show_centers
        ])

        # Create tabs (store reference for later updates)
        self.main_tab = Tab()
        self.main_tab.children = [global_tab, particles_tab, field_tab]
        self.main_tab.set_title(0, '⚙️ Global')
        self.main_tab.set_title(1, '🔘 Particles')
        self.main_tab.set_title(2, '📡 Incident EM wave')

        # Main layout - controls above, output below
        controls = VBox([
            Label('🔬 Interactive Nanocluster Builder',
                  layout={'width': '100%'}),
            self.main_tab,
            self.update_btn
        ])

        # Stack vertically instead of horizontally
        display(VBox([controls, self.output]))

    def on_update_click(self, b):
        """Handle update button click."""
        self.update_plot()

    def get_current_parameters(self):
        """Extract current parameter values from widgets."""
        n = self.n_particles.value
        positions, diameters, materials = [], [], []

        for i in range(n):
            if i < len(self.particle_widgets):
                widgets = self.particle_widgets[i].children
                x = widgets[1].value
                y = widgets[2].value
                z = widgets[3].value
                d = widgets[4].value
                mat = widgets[5].value

                positions.append(np.array([x, y, z]))
                diameters.append(d)
                materials.append(mat)

        self.current_positions = positions
        self.current_diameters = diameters
        self.current_materials = materials

        return positions, diameters, materials

    def check_overlaps(self, positions, diameters):
        """
        Check for overlapping nanoparticles.

        Returns
        -------
        list of tuples
            List of (i, j, overlap_distance) for overlapping pairs
        """
        overlaps = []
        n = len(positions)

        for i in range(n):
            for j in range(i + 1, n):
                # Distance between centers
                distance = np.linalg.norm(positions[i] - positions[j])

                # Sum of radii
                min_distance = (diameters[i] + diameters[j]) / 2.0

                # Check for overlap (with small tolerance for surface contact)
                if distance < min_distance - 0.01:  # 0.01 nm tolerance
                    overlap = min_distance - distance
                    overlaps.append((i, j, overlap))

        return overlaps

    def compute_e_vector(self, k_vec, angle_degrees):
        """
        Compute E-vector from k-vector and polarization angle.

        The E-vector is constructed in the plane perpendicular to k by rotating
        around k by the specified angle.

        Parameters
        ----------
        k_vec : array-like
            Normalized k-vector (propagation direction)
        angle_degrees : float
            Polarization angle in degrees (0-360)

        Returns
        -------
        np.ndarray
            Normalized E-vector perpendicular to k
        """
        # Convert angle to radians
        theta = np.radians(angle_degrees)

        # Find two orthonormal vectors perpendicular to k
        # First perpendicular vector: choose based on which component of k is smallest
        k_normalized = bcm.v_normalize(k_vec)

        if abs(k_normalized[2]) < 0.9:
            # k is not close to z-axis, use z as reference
            v1 = np.cross(k_normalized, [0, 0, 1])
        else:
            # k is close to z-axis, use x as reference
            v1 = np.cross(k_normalized, [1, 0, 0])

        v1 = bcm.v_normalize(v1)

        # Second perpendicular vector: cross product of k and v1
        v2 = np.cross(k_normalized, v1)
        v2 = bcm.v_normalize(v2)

        # Construct E-vector by rotating in the v1-v2 plane
        e_vec = np.cos(theta) * v1 + np.sin(theta) * v2

        return bcm.v_normalize(e_vec)

    def update_plot(self):
        """Update the 3D visualization in a separate browser window."""
        positions, diameters, materials = self.get_current_parameters()

        # Compute k-vector (propagation direction)
        k_vec = bcm.v_normalize([self.k_x.value, self.k_y.value, self.k_z.value])

        # Compute E-vector from polarization angle (always perpendicular to k)
        e_vec = self.compute_e_vector(k_vec, self.pol_angle.value)

        # Check for overlaps
        overlaps = self.check_overlaps(positions, diameters)

        with self.output:
            clear_output(wait=True)

            # Display overlap warnings
            if overlaps:
                print("\n" + "⚠️ " * 35)
                print("❌ WARNINGS DETECTED!")
                print("⚠️ " * 35 + "\n")
                print("🔴 NANOPARTICLE OVERLAP:")
                print("   The following particle pairs are overlapping:")
                print("   (This will cause physics simulations to fail!)\n")
                for i, j, overlap in overlaps:
                    print(f"   • Particle {i+1} ↔ Particle {j+1}: overlap = {overlap:.3f} nm")
                print()
                print("⚠️ " * 35 + "\n")

            # Create Plotly figure
            fig = go.Figure()

            # Create list of overlapping particle indices
            overlapping_indices = set()
            for i, j, _ in overlaps:
                overlapping_indices.add(i)
                overlapping_indices.add(j)

            # Add spheres
            resolution = self.sphere_resolution.value
            for i, (pos, diam, mat) in enumerate(zip(positions, diameters, materials), 1):
                radius = diam / 2.0
                x, y, z = create_sphere_mesh(pos, radius, resolution)

                # Always use material color
                color = MATERIALS[mat]['color']

                # Add warning prefix if overlapping
                if (i - 1) in overlapping_indices:
                    name_prefix = '⚠️ '
                    hover_warning = '<br><b>⚠️ OVERLAPPING!</b>'
                    # Add red glow effect by adjusting lighting
                    ambient = 0.8  # Brighter for overlapping
                    diffuse = 0.5
                else:
                    name_prefix = ''
                    hover_warning = ''
                    ambient = 0.6
                    diffuse = 0.8

                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f'{name_prefix}{mat} #{i}',
                    hovertemplate=(
                        f'<b>{mat} Sphere #{i}</b><br>'
                        f'Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) nm<br>'
                        f'Diameter: {diam:.2f} nm{hover_warning}<br>'
                        '<extra></extra>'
                    ),
                    lighting=dict(ambient=ambient, diffuse=diffuse, specular=0.4, roughness=0.3),
                    lightposition=dict(x=100, y=200, z=150)
                ))

                # Add particle number label - place at sphere center for visibility
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='text',
                    text=[f'<b>#{i}</b>'],
                    textfont=dict(
                        size=18,
                        color='white',
                        family='Arial Black'
                    ),
                    textposition='middle center',
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Add center markers
                if self.show_centers.value:
                    # Use red color for overlapping particles
                    center_color = 'red' if (i - 1) in overlapping_indices else 'black'
                    center_size = 8 if (i - 1) in overlapping_indices else 4
                    fig.add_trace(go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]],
                        mode='markers',
                        marker=dict(size=center_size, color=center_color, symbol='x' if (i - 1) in overlapping_indices else 'diamond'),
                        name=f'Center {i}',
                        showlegend=False,
                        hoverinfo='skip'
                    ))

            # Add field vectors
            arrow_length_total = 0
            if self.show_field.value:
                pos_array = np.array(positions)
                radii_array = np.array(diameters) / 2.0

                mins = pos_array.min(axis=0) - radii_array.max()
                maxs = pos_array.max(axis=0) + radii_array.max()

                # Place arrows at corner with proper clearance
                scene_span = np.max(maxs - mins)
                arrow_length = 0.28 * scene_span
                arrow_length_total = arrow_length  # Store for scene bounds adjustment

                # Position at corner with clearance
                arrow_origin = np.array([mins[0], maxs[1], mins[2]])

                # k vector (red)
                for arrow in create_arrow(arrow_origin, k_vec, arrow_length, 'red', 'k-vector'):
                    fig.add_trace(arrow)

                # E vector (blue)
                E_hat = e_vec / np.linalg.norm(e_vec)
                for arrow in create_arrow(arrow_origin, E_hat, arrow_length*0.85, 'blue', 'E-vector'):
                    fig.add_trace(arrow)

                # H vector (green)
                H_hat = np.cross(k_vec, E_hat)
                for arrow in create_arrow(arrow_origin, H_hat, arrow_length*0.85, 'green', 'H-vector'):
                    fig.add_trace(arrow)

            # Layout - ensure equal aspect ratio for spherical appearance
            pos_array = np.array(positions)
            radii_array = np.array(diameters) / 2.0

            # Calculate bounding box with equal padding
            center = pos_array.mean(axis=0)
            max_extent = max(
                (pos_array[:, 0].max() - pos_array[:, 0].min()),
                (pos_array[:, 1].max() - pos_array[:, 1].min()),
                (pos_array[:, 2].max() - pos_array[:, 2].min())
            ) + 2 * radii_array.max()

            # Make cubic scene with equal ranges
            # Add extra space if arrows are shown
            arrow_padding = arrow_length_total * 1.2 if self.show_field.value and arrow_length_total > 0 else 0
            half_range = max_extent * 0.65 + arrow_padding
            x_range = [center[0] - half_range, center[0] + half_range]
            y_range = [center[1] - half_range, center[1] + half_range]
            z_range = [center[2] - half_range, center[2] + half_range]

            # Update title with overlap warning if needed
            if overlaps:
                title_text = f'<b>⚠️ Nanocluster Geometry (N={len(positions)}) - OVERLAP DETECTED!</b>'
                title_color = 'red'
            else:
                title_text = f'<b>Nanocluster Geometry (N={len(positions)}) ✓</b>'
                title_color = 'black'

            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color=title_color)
                ),
                scene=dict(
                    xaxis=dict(title='x (nm)', range=x_range,
                              backgroundcolor="rgb(245, 245, 250)",
                              gridcolor="white", gridwidth=2),
                    yaxis=dict(title='y (nm)', range=y_range,
                              backgroundcolor="rgb(245, 245, 250)",
                              gridcolor="white", gridwidth=2),
                    zaxis=dict(title='z (nm)', range=z_range,
                              backgroundcolor="rgb(245, 245, 250)",
                              gridcolor="white", gridwidth=2),
                    aspectmode='cube',  # Force equal aspect ratio
                    aspectratio=dict(x=1, y=1, z=1),  # Explicit 1:1:1 ratio
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                showlegend=True,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
                hovermode='closest',
                width=1000,
                height=800
            )

            # Show the figure based on display mode
            if self.display_mode == 'inline':
                # Display inline in notebook with full interactivity
                fig.show()
            elif self.display_mode == 'browser':
                # Try to open in browser
                try:
                    fig.show(renderer='browser')
                except Exception as e:
                    print(f"⚠️  Could not open browser: {e}")
                    print("   Displaying inline instead...")
                    fig.show()
            else:  # 'auto'
                # Try browser first, fall back to inline
                try:
                    fig.show(renderer='browser')
                except Exception:
                    fig.show()

            # Print summary
            print("\n" + "="*70)
            print("NANOCLUSTER CONFIGURATION SUMMARY")
            print("="*70)
            print(f"Number of particles: {len(positions)}")

            # Overlap status
            if overlaps:
                print(f"\n⚠️  OVERLAP STATUS: ❌ {len(overlaps)} overlap(s) detected!")
                print("   ⚠️  Physics simulations will FAIL with overlapping particles!")
            else:
                print(f"\n✓  OVERLAP STATUS: ✅ No overlaps detected - geometry is valid!")

            print(f"\nIncident field:")
            print(f"  k-vector (propagation):     [{k_vec[0]:.3f}, {k_vec[1]:.3f}, {k_vec[2]:.3f}]")
            print(f"  Polarization angle:         {self.pol_angle.value:.1f}°")
            print(f"  E-vector (polarization):    [{e_vec[0]:.3f}, {e_vec[1]:.3f}, {e_vec[2]:.3f}]")
            print(f"  H-vector (k × E):           [{np.cross(k_vec, e_vec)[0]:.3f}, {np.cross(k_vec, e_vec)[1]:.3f}, {np.cross(k_vec, e_vec)[2]:.3f}]")
            print(f"  Verification: k · E = {np.dot(k_vec, e_vec):.6f} (should be ~0)")
            print(f"\nParticles:")
            for i, (pos, diam, mat) in enumerate(zip(positions, diameters, materials), 1):
                print(f"  {i}. {mat:12s} D={diam:5.1f} nm  "
                      f"pos=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) nm")
            print("="*70)
            if self.display_mode == 'inline':
                print("\n💡 Interactive 3D plot displayed above (inline mode)")
            else:
                print("\n💡 Interactive 3D plot opened")
            print("   • Drag to rotate  • Scroll to zoom  • Shift+drag to pan")
            print()

    def export_bcm_objects(self, eps_drude, lmax=10):
        """
        Export current configuration as BCM_objects list.

        Parameters
        ----------
        eps_drude : callable
            Dielectric function
        lmax : int
            Maximum spherical harmonic index

        Returns
        -------
        list of BCMObject
            Ready to use in BCM simulations
        """
        positions, diameters, materials = self.get_current_parameters()

        BCM_objects = []
        for i, (pos, diam, mat) in enumerate(zip(positions, diameters, materials), 1):
            obj = BCMObject(
                label=f'Sphere{i}_{mat}',
                diameter=diam,
                lmax=lmax,
                eps=eps_drude,
                position=pos
            )
            BCM_objects.append(obj)

        return BCM_objects

    def export_efield(self, E0=1):
        """
        Export current EField configuration.

        Parameters
        ----------
        E0 : float
            Electric field amplitude (V/nm)

        Returns
        -------
        EField
            Ready to use in BCM simulations
        """
        k_vec = bcm.v_normalize([self.k_x.value, self.k_y.value, self.k_z.value])
        e_vec = self.compute_e_vector(k_vec, self.pol_angle.value)

        return EField(E0=E0, k_hat=k_vec, e_hat=e_vec)


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
    out_path,
    title_prefix: str = "",
    fps: int = 1,
):
    """
    Build a GIF where each frame is a 1x2 grid:
      [ Absorption vs hv ]  |  [ Hot-carrier distribution ]

    Each frame corresponds to one gap value Delta at a fixed lifetime tau.
    """
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    n_levels = len(E_all)
    if any(len(np.asarray(arr)) != n_levels for arr in Te_tau_frames):
        raise ValueError("Length mismatch between Te_tau_frames and E_all.")
    if any(len(np.asarray(arr)) != n_levels for arr in Th_tau_frames):
        raise ValueError("Length mismatch between Th_tau_frames and E_all.")

    to_ps = 1000.0   # fs -> ps
    dt    = 1.0 / max(fps, 1)

    with imageio.get_writer(out_path, mode="I", duration=dt) as writer:
        for k, gap_nm in enumerate(gap_values_nm):
            gap_nm = float(gap_nm)
            hv     = float(hv_frames[k]) if k < len(hv_frames) else 0.0
            if hv <= 0.0:
                hv_pos = [h for h in hv_frames if h > 0]
                hv     = float(np.mean(hv_pos)) if hv_pos else 1e-6

            Te     = np.asarray(Te_tau_frames[k], float) / 1000.0
            Th     = np.asarray(Th_tau_frames[k], float) / 1000.0
            Te_raw = np.asarray(Te_raw_frames[k], float)
            Th_raw = np.asarray(Th_raw_frames[k], float)

            x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)
            scale  = to_ps / hv
            Te_x  *= scale
            Th_x  *= scale

            mask_e = (x >= EF) & (x <= EF + gap_nm)
            mask_h = (x <= EF) & (x >= EF - gap_nm)

            Qabs     = np.asarray(Qabs_frames[k], float)
            hv_lines = hv_lines_per_delta[k]

            fig, (ax_abs, ax_hc) = plt.subplots(1, 2, figsize=(13, 4.5))

            # -- Left: Absorption --
            ax_abs.plot(energy_eV, Qabs, 'k')
            for hv_line in hv_lines:
                ax_abs.axvline(hv_line, ls=':', lw=0.8, color='gray', alpha=0.6)
            ax_abs.axvline(hv, ls='--', lw=1.2, color='red', alpha=0.95,
                           label=r'resonance $h\nu$')
            ax_abs.set_xlabel('Photon energy (eV)')
            ax_abs.set_ylabel('Absorption efficiency')
            ax_abs.set_title(rf'Absorption · $\Delta={gap_nm:.2f}$ nm')
            ax_abs.grid(True, ls=':')
            ax_abs.legend(loc='upper right')

            # -- Right: Hot carriers --
            ax2 = ax_hc.twinx()
            ax_hc.fill_between((x - EF)[mask_e], Te_x[mask_e],
                               color='r', alpha=0.38, label='Electrons (dens.)')
            ax_hc.fill_between((x - EF)[mask_h], Th_x[mask_h],
                               color='b', alpha=0.38, label='Holes (dens.)')

            bar_w = 2.0e-2
            ax2.bar(E_all - EF, Te_raw * to_ps, width=bar_w,
                    color='firebrick', alpha=0.9, label='Electrons')
            ax2.bar(E_all - EF, Th_raw * to_ps, width=bar_w,
                    color='royalblue', alpha=0.9, label='Holes')

            ax_hc.axvline(0.0,  ls='--', lw=1,   color='k',    alpha=0.5)
            ax_hc.axvline(+hv,  ls='--', lw=1,   color='gray', alpha=0.6)
            ax_hc.axvline(-hv,  ls='--', lw=1,   color='gray', alpha=0.6)
            ax_hc.set_xlim(-gap_nm, +gap_nm)
            ax_hc.set_xlabel('Hot carrier energy relative to Fermi level (eV)')
            ax_hc.set_ylabel(
                r'Rate density $[10^{-3}\,\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
            ax2.set_ylabel(r'Rate per particle $[\mathrm{ps}^{-1}]$')
            ax_hc.set_ylim(0.0, 1.05 * max(Te_x[mask_e].max(initial=0.0),
                                            Th_x[mask_h].max(initial=0.0)))
            ax2.set_ylim(0.0, 3.0 * to_ps * max(Te_raw.max(initial=0.0),
                                                  Th_raw.max(initial=0.0)))
            h1, l1 = ax_hc.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax_hc.legend(h1 + h2, l1 + l2, loc='upper right')

            fig.suptitle(
                rf'{title_prefix}'
                rf'\n$\Delta={gap_nm:.2f}\,\mathrm{{nm}},\ h\nu={hv:.2f}\,\mathrm{{eV}},'
                rf'\ \tau={tau_fs/1000:.2f}\,\mathrm{{ps}}$'
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
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
    out_path,
    title_prefix: str = "",
    fps: int = 1,
):
    """
    Build a GIF with one absorption panel + up to 3 hot-carrier distributions
    (one per tracked resonance) for each gap step Delta.
    """
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resonances_data:
        return

    resonances_data  = resonances_data[:3]   # at most 3 panels
    hv_static_list   = [res["center_eV"] for res in resonances_data]
    n_res            = len(resonances_data)
    colors_static    = ["tab:red", "tab:blue", "tab:green"]

    E_all  = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    to_ps  = 1000.0   # fs -> ps
    dt     = 1.0 / max(fps, 1)

    with imageio.get_writer(out_path, mode="I", duration=dt) as writer:
        for k, gap_nm in enumerate(gap_values_nm):
            gap_nm = float(gap_nm)

            fig, axes = plt.subplots(n_res + 1, 1,
                                     figsize=(8, 3.0 * (n_res + 1)),
                                     sharex=True)
            if n_res == 1:
                axes = [axes]
            ax_abs  = axes[0]
            ax_list = axes[1:]

            # -- Absorption panel --
            Qabs        = np.asarray(Qabs_frames[k], float)
            hv_lines_dyn = hv_lines_per_delta[k]

            ax_abs.plot(energy_eV, Qabs, "k")
            for hv_line in hv_lines_dyn:
                ax_abs.axvline(hv_line, ls=":", lw=0.8, color="gray", alpha=0.6)
            for i, hv_static in enumerate(hv_static_list):
                ax_abs.axvline(hv_static, ls="--", lw=1.4,
                               color=colors_static[i % len(colors_static)],
                               alpha=0.9,
                               label=rf"res {i+1}: {hv_static:.2f} eV")
            ax_abs.set_ylabel(r"$Q_{abs}$")
            ax_abs.set_title(rf"Absorption · $\Delta={gap_nm:.2f}\,\mathrm{{nm}}$")
            ax_abs.grid(True, ls=":")
            ax_abs.legend(loc="upper right")

            # -- One HC panel per resonance --
            for i_res, res in enumerate(resonances_data):
                ax    = ax_list[i_res]
                hv_dyn = float(res["hv_frames"][k]) if k < len(res["hv_frames"]) else 1e-6
                if hv_dyn <= 0.0:
                    hv_dyn = 1e-6

                Te = np.asarray(res["Te_tau_frames"][k], float) / 1000.0
                Th = np.asarray(res["Th_tau_frames"][k], float) / 1000.0

                x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)
                scale  = to_ps / hv_dyn
                Te_x  *= scale
                Th_x  *= scale

                mask_e = (x >= EF) & (x <= EF + gap_nm)
                mask_h = (x <= EF) & (x >= EF - gap_nm)

                ax.fill_between((x - EF)[mask_e], Te_x[mask_e],
                                alpha=0.4, label=r"e$^-$")
                ax.fill_between((x - EF)[mask_h], Th_x[mask_h],
                                alpha=0.4, label=r"h$^+$")

                hv_static = hv_static_list[i_res]
                c = colors_static[i_res % len(colors_static)]
                ax.axvline(+hv_static, ls="--", lw=1.2, color=c, alpha=0.9)
                ax.axvline(-hv_static, ls="--", lw=1.2, color=c, alpha=0.9)

                ax.set_ylabel(
                    rf"res {i_res+1}  "
                    rf"$[10^{{-3}}\,\mathrm{{eV}}^{{-1}}\mathrm{{ps}}^{{-1}}\mathrm{{nm}}^{{-3}}]$"
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
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(frame)
            plt.close(fig)


def launch_builder(display_mode='inline'):
    """
    Launch the interactive nanocluster builder.

    Parameters
    ----------
    display_mode : str, optional
        Display mode: 'inline' (default), 'browser', or 'auto'

    Returns
    -------
    NanoclusterBuilder
        The builder instance for exporting configurations
    """
    print("🚀 Launching Interactive Nanocluster Builder...")
    if display_mode == 'inline':
        print("💡 3D plots will display inline in the notebook")
    elif display_mode == 'browser':
        print("💡 3D plots will open in your browser")
    else:
        print("💡 3D plots will try browser, fallback to inline")
    print()
    return NanoclusterBuilder(display_mode=display_mode)
