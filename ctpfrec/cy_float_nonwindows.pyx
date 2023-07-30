#cython: language_level=3, legacy_implicit_noexcept=True
import ctypes
from libc.math cimport HUGE_VALL, HUGE_VALL as LD_HUGE_VAL

obj_ind_type = ctypes.c_size_t
ctypedef size_t ind_type
ctypedef long double long_double_type
obj_long_double_type = ctypes.c_longdouble

include "cy_float.pxi"
