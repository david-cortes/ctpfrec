from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

if platform.system() == "Windows":
	ext_mod = Extension("ctpfrec.cy",
                             sources=["ctpfrec/cy.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['/openmp', '/O2'])
else:
	ext_mod = Extension("ctpfrec.cy",
                             sources=["ctpfrec/cy.pyx"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-fopenmp', '-O2'],
                             extra_link_args=['-fopenmp'])
### Comment: -Ofast gives worse speed than -O2 or -O3

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
  version = '0.1',
  description = 'Collaborative topic Poisson factorization for recommender systems',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/ctpfrec',
  download_url = 'https://github.com/david-cortes/ctpfrec/archive/0.1.tar.gz',
  keywords = ['collaborative', 'topic', 'modeling', 'poisson', 'probabilistic', 'non-negative', 'factorization',
              'variational inference', 'collaborative filtering', 'cold-start'],
  classifiers = [],

  cmdclass = {'build_ext': build_ext},
  ext_modules = [ext_mod,
    ]
)
