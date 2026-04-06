"""
Plytrons - Tools for computing hot carrier distributions in metallic nanoparticle clusters.
"""

# Import interactive builder for easy access
try:
    from .plot_utils import launch_builder, NanoclusterBuilder, MATERIALS, MEDIA
except ImportError:
    # Plotly / ipywidgets not installed
    pass

__version__ = "0.1.0"
