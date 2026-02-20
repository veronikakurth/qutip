# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
from qutip import settings
from qutip.core.data cimport CSR, Dia, Dense

cdef extern from "<complex>" namespace "std":
    double abs(double complex z) nogil

cdef inline bint isclose(double complex z, double atol) nogil:
    return abs(z.real) <= atol and abs(z.imag) <= atol

# This module is meant to be accessed by dot-access (e.g. mean.mean_csr).
__all__ = []

cpdef double complex mean_csr(CSR matrix) noexcept nogil:
    cdef size_t nnz, ptr, nnz_corrected = 0
    cdef double complex mean = 0
    cdef double atol

    with gil:
        atol = settings.core['atol']

    nnz = matrix.row_index[matrix.shape[0]] # TODO: since close to zero values may be stored in CSR, filter them out and then think if it's a good solution

    if nnz == 0:
        return 0.0

    for ptr in range(nnz):
        if isclose(matrix.data[ptr], atol=atol):
            continue
        mean += matrix.data[ptr]
        nnz_corrected += 1

    if nnz_corrected == 0:
        return 0.0

    return mean / nnz_corrected

cdef inline int int_max(int a, int b) noexcept nogil:
    return a if a > b else b

cdef inline int int_min(int a, int b) noexcept nogil:
    return a if a < b else b

cpdef double complex mean_dia(Dia matrix) noexcept nogil:
    cdef int offset, diag, start, end, col=1
    cdef double complex mean = 0
    cdef size_t nnz = 0
    cdef double atol

    with gil:
        atol = settings.core['atol']

    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = int_max(0, offset)
        end = int_min(matrix.shape[1], matrix.shape[0] + offset)
        if end < start:
            continue

        for col in range(start, end):
            cur_el = matrix.data[diag * matrix.shape[1] + col]
            if isclose(cur_el, atol=atol):
                continue
            mean += cur_el
            nnz += 1

    if nnz == 0:
        return 0.0
    return mean / nnz


cpdef double complex mean_dense(Dense matrix) noexcept nogil:
    cdef size_t ptr, nnz = 0
    cdef double complex mean = 0, cur
    cdef double atol

    with gil:
        atol = settings.core['atol']

    for ptr in range(matrix.shape[0] * matrix.shape[1]):
        cur = matrix.data[ptr]

        if isclose(cur, atol=atol):
            continue
        mean += cur
        nnz += 1

    if nnz == 0:
        return 0.0
    return mean / nnz

cpdef double mean_abs_csr(CSR matrix) noexcept nogil:
    cdef size_t nnz, ptr, nnz_corrected = 0
    cdef double mean = 0
    cdef double atol

    with gil:
        atol = settings.core['atol']

    nnz = matrix.row_index[matrix.shape[0]]

    if nnz == 0:
        return 0.0

    for ptr in range(nnz):
        if isclose(matrix.data[ptr], atol=atol):
            continue
        else:
            mean += abs(matrix.data[ptr])
            nnz_corrected += 1

    if nnz_corrected == 0:
        return 0.0

    return mean / nnz_corrected

cpdef double mean_abs_dia(Dia matrix) noexcept nogil:
    cdef int offset, diag, start, end, col=1
    cdef double mean_abs = 0
    cdef size_t nnz = 0
    cdef double atol

    with gil:
        atol = settings.core['atol']

    for diag in range(matrix.num_diag):
        offset = matrix.offsets[diag]
        start = int_max(0, offset)
        end = int_min(matrix.shape[1], matrix.shape[0] + offset)

        if end < start:
            continue

        for col in range(start, end):
            cur_el = matrix.data[diag * matrix.shape[1] + col]
            if isclose(cur_el, atol=atol):
                continue
            mean_abs += abs(matrix.data[diag * matrix.shape[1] + col])
            nnz += 1
    if nnz == 0:
        return 0.0
    return mean_abs / nnz

cpdef double mean_abs_dense(Dense matrix) noexcept nogil:
    cdef size_t ptr, nnz = 0
    cdef double mean_abs = 0, cur
    cdef double atol

    with gil:
        atol = settings.core['atol']
    
    for ptr in range(matrix.shape[0] * matrix.shape[1]):
        cur = abs(matrix.data[ptr])
        if isclose(cur, atol=atol):
            continue
        mean_abs += cur
        nnz += 1
    if nnz == 0:
        return 0.0
    return mean_abs / nnz

from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

mean = _Dispatcher(
            _inspect.Signature([
                _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
            ]),
            name='mean_nonzero',
            module=__name__,
            inputs=('matrix',),
)
mean.__doc__ =\
        """
        Adapted mean value: compute the mean value of non-zero entries of a matrix.
        """
mean.add_specialisations([
    (Dense, mean_dense),
    (Dia, mean_dia),
    (CSR, mean_csr),
], _defer=True)

mean_abs = _Dispatcher(
            _inspect.Signature([
                _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_ONLY),
            ]),
            name='mean_abs_nonzero',
            module=__name__,
            inputs=('matrix',),
)
mean_abs.__doc__ =\
        """
        Adapted mean value: \
        compute the mean value of absolute values of non-zero entries of a matrix.
        """
mean_abs.add_specialisations([
    (Dense, mean_abs_dense),
    (Dia, mean_abs_dia),
    (CSR, mean_abs_csr),
], _defer=True)
