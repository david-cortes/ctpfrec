from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
  def build_extensions(self):
    c = self.compiler.compiler_type
    # TODO: add entries for intel's ICC
    if c == 'msvc': # visual studio
      for e in self.extensions:
        e.extra_compile_args = ['/openmp', '/O2']
    else: # gcc and clang
      for e in self.extensions:
        e.extra_compile_args = ['-fopenmp', '-O3']
        e.extra_link_args = ['-fopenmp']
        ### Comment: -Ofast gives worse speed than -O2 or -O3
    build_ext.build_extensions(self)

setup(
  name = 'ctpfrec',
  packages = ['ctpfrec'],
  install_requires=[
   'pandas>=0.21',
   'numpy',
   'scipy',
   'cython',
   'hpfrec'
],
  version = '0.1.1',
  description = 'Collaborative topic Poisson factorization for recommender systems',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/ctpfrec',
  keywords = ['collaborative', 'topic', 'modeling', 'poisson', 'probabilistic', 'non-negative', 'factorization',
              'variational inference', 'collaborative filtering', 'cold-start'],
  classifiers = [],

  cmdclass = {'build_ext': build_ext_subclass},
  ext_modules = [Extension("ctpfrec.cy", sources=["ctpfrec/cy.pyx"], include_dirs=[numpy.get_include()]),
    ]
)
