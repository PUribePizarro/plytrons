from setuptools import setup, find_packages

setup(
    name='plytrons',
    version='0.1.0',
    description='Tools for computing hot carrier distributions in clustered metallic nanoparticles using semi-analytical models',
    long_description=(
        'Plytrons is a Python library for modeling plasmon-induced hot carrier '
        'distributions in metallic nanoparticle clusters. It combines the spherical '
        'quantum well model, Fermi\'s golden rule, and plasmon hybridization theory to '
        'provide efficient semi-analytical solutions for electromagnetic excitation and '
        'carrier generation.'
    ),
    author='Francisco Ramírez, Pablo Uribe',
    author_email='fvr@alumni.cmu.edu',
    url='https://github.com/PanxoPanza/plytrons.git',
    packages=find_packages(),
    install_requires=[
        'numpy==1.22.4',
        'scipy==1.7.3',
        'numba',
        'numba-scipy==0.3.1',
        'matplotlib',
        'plotly',
        'ipywidgets',
        'tqdm',
        'imageio',
    ],
    extras_require={
        'dev': ['pytest'],
        'docs': ['jupyter', 'notebook'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.7',
)
