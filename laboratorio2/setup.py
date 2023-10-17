from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("sharpen_cython.pyx"),
    extra_compile_args=['-arch', 'x86_64'],
)
