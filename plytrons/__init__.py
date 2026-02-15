"""
Plytrons - Tools for computing hot carrier distributions in metallic nanoparticle clusters.
"""

# Import interactive builder for easy access
try:
    from .interactive_builder import launch_builder, NanoclusterBuilder, MATERIALS
except ImportError:
    # Plotly not installed
    pass

__version__ = "0.1.0"
