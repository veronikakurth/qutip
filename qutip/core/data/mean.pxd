#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Dia

# TODO: verify signatures for Cython functions - are except/nogil keywords necessary?
cpdef double complex mean_csr(CSR matrix) noexcept nogil
cpdef double complex mean_dia(Dia matrix) noexcept nogil
cpdef double complex mean_dense(Dense matrix) noexcept nogil

cpdef double mean_abs_csr(CSR matrix) noexcept nogil
cpdef double mean_abs_dia(Dia matrix) noexcept nogil
cpdef double mean_abs_dense(Dense matrix) noexcept nogil

cdef inline int int_max(int a, int b) nogil:
    # Name collision between the ``max`` builtin and norm.max
    return b if b > a else a
