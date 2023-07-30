#cython: language_level=3, legacy_implicit_noexcept=True
import ctypes
from libc.math cimport HUGE_VAL, HUGE_VAL as LD_HUGE_VAL

obj_ind_type = ctypes.c_longlong
ctypedef long long ind_type
ctypedef double long_double_type
obj_long_double_type = ctypes.c_double

include "cy_float.pxi"
