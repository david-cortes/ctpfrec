import ctypes
from libc.math cimport HUGE_VAL

obj_ind_type = ctypes.c_longlong
ctypedef long long ind_type
ctypedef double long_double_type
obj_long_double_type = ctypes.c_double
LD_HUGE_VAL = HUGE_VAL

include "cy_double.pxi"
