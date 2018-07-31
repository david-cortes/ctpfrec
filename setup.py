from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True



cython_loops = Extension('cython_loops', sources=['ctpfrec/cython_loops.pyx'], include_dirs=[numpy.get_include()], extra_compile_args=['-fopenmp', '-O2'], extra_link_args=['-fopenmp'])

setup(
    ext_modules = cythonize(cython_loops, annotate=True)
)