
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize('cy_tree.pyx'), include_dirs=[np.get_include()])


# build via: python3 cy_tree_setup.py build_ext --inplace
