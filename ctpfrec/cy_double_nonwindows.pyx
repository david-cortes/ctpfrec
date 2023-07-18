import ctypes
from libc.math cimport HUGE_VALL

obj_ind_type = ctypes.c_size_t
ctypedef size_t ind_type
ctypedef long double long_double_type
obj_long_double_type = ctypes.c_longdouble
LD_HUGE_VAL = HUGE_VALL

include "cy_float.pxi"
