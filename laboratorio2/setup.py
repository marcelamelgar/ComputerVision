from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("sharpen_cython.pyx"),
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-arch', 'x86_64'],
)
