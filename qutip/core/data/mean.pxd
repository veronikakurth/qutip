#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Dia

cpdef double complex mean_csr(CSR matrix) noexcept
cpdef double complex mean_dia(Dia matrix) noexcept nogil
cpdef double complex mean_dense(Dense matrix) noexcept

cpdef double mean_abs_csr(CSR matrix) noexcept
cpdef double mean_abs_dia(Dia matrix) noexcept nogil
cpdef double mean_abs_dense(Dense matrix) noexcept


