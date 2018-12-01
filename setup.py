from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

## Note: As of the end of 2018, MSVC is still stuck with OpenMP 2.0 (released 2002), which does not support
## parallel for loops with unsigend iterators. If you are using a different compiler, this part can be safely removed
if platform.system() == "Windows":
  import re
  fout = open("temp1.txt", "w")
  with open("ctpfrec\\cy.pyx", "r") as fin:
      for line in fin:
          fout.write(re.sub("size_t([^\w])", "long\\1", line))
  fout.close()
  fout = open("temp2.txt", "w")
  with open("ctpfrec\\__init__.py", "r") as fin:
      for line in fin:
          fout.write(re.sub("size_t([^\w])", "long\\1", line))
  fout.close()

  fout = open("ctpfrec\\cy.pyx", "w")
  with open("temp1.txt", "r") as fin:
      for line in fin:
          fout.write(line)
  fout.close()
  fout = open("ctpfrec\\__init__.py", "w")
  with open("temp2.txt", "r") as fin:
      for line in fin:
          fout.write(line)
  fout.close()

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
  version = '0.1.3.3',
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
