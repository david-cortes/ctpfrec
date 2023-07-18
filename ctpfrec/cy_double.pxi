#cython: language_level=3
from scipy.linalg.cython_blas cimport ddot
from hpfrec import cython_loops_double as cython_loops
import ctypes

from libc.math cimport log, exp, HUGE_VAL, HUGE_VALL

c_real_t = ctypes.c_double
ctypedef double real_t

ctypedef real_t (*blas_dot)(int*, real_t*, int*, real_t*, int*) nogil
cdef blas_dot tdot = ddot

ctypedef real_t (*real_t_fun)(real_t) nogil
cdef real_t_fun exp_t = exp
cdef real_t_fun log_t = log

cdef real_t HUGE_VAL_T = HUGE_VAL

include "cy.pxi"
