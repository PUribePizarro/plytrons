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

    elif preset_name == 'Tetramer':
        return [
            np.array([0, d_c, d_c]),
            np.array([0, d_c, -d_c]),
            np.array([0, -d_c, d_c]),
            np.array([0, -d_c, -d_c])
        ]

    elif preset_name == 'Pentamer (planar)':
        N = 5
        return [
            np.array([
                0,
                (d_c/(2*np.sin(np.pi/N)))*np.cos(np.pi/2 + 2*np.pi/N*i),
                (d_c/(2*np.sin(np.pi/N)))*np.sin(np.pi/2 + 2*np.pi/N*i)
            ]) for i in range(N)
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

    return [np.array([0, 0, 0])]


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
            options=['Custom', 'Monomer', 'Dimer', 'Trimer', 'Tetramer',
                     'Pentamer (planar)', 'Hexamer (ring)', 'Heptamer (ring + center)'],
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
