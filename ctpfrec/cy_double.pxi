#cython: language_level=3
from scipy.linalg.cython_blas cimport ddot, ddot as tdot
from hpfrec import cython_loops_double as cython_loops
import ctypes

from libc.math cimport log, exp, HUGE_VAL, HUGE_VALL
from libc.math cimport log as log_t, exp as exp_t, HUGE_VAL as HUGE_VAL_T

c_real_t = ctypes.c_double
ctypedef double real_t

include "cy.pxi"
