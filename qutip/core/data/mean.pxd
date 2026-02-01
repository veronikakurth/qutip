from qutip.core.data cimport CSR, Dense, Dia, Data

# TODO: verify signatures for Cython functions - are except/nogil keywords necessary?
cpdef double complex mean_csr(CSR matrix) nogil
cpdef double complex mean_dia(Dia matrix) nogil
cpdef double complex mean_dense(Dense matrix) nogil

cpdef double mean_abs_csr(CSR matrix) nogil
cpdef double mean_abs_dia(Dia matrix) nogil
cpdef double mean_abs_dense(Dense matrix) nogil
