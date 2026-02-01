from qutip.core.data cimport CSR, Dia
from libc.math cimport fabs
from scipy.linalg cimport cython_blas as blas

cpdef double complex mean_csr(CSR matrix) nogil:
  cdef size_t nnz, ptr
  cdef double complex mean = 0

  nnz = matrix.row_index[matrix.shape[0]]

  if nnz == 0:
    return 0.0

  for ptr in range(nnz):
    mean += matrix.data[ptr]
  
  mean = mean / nnz
  return mean


cpdef double complex mean_dia(Dia matrix) nogil:
  cdef int offset, diag, start, end, col=1
  cdef double complex mean = 0
  cdef size_t nnz = 0

  for diag in range(matrix.num_diag):
      offset = matrix.offsets[diag]
      start = int_max(0, offset)
      end = min(matrix.shape[1], matrix.shape[0] + offset)
      for col in range(start, end):
          mean += matrix.data[diag * matrix.shape[1] + col]
          nnz += 1
  if nnz == 0:
    return 0.0
  
  mean = mean/nnz
  return mean


cpdef double complex mean_dense(Dense matrix) nogil:
  cdef size_t ptr, nnz = 0
  cdef double complex mean = 0, cur
  
  for ptr in range(matrix.shape[0] * matrix.shape[1]):
    cur = matrix.data[ptr]

    if cur == 0.0:
      continue
    
    mean += cur
    nnz += 1
  
  mean /= nnz
  return mean

cpdef double mean_abs_csr(CSR matrix) nogil:
  cdef int nnz, inc = 1
  nnz = matrix.row_index[matrix.shape[0]]
  
  if nnz == 0:
    return 0.0

  return blas.dzasum(&nnz, &matrix.data[0], &inc) / nnz
  

cpdef double mean_abs_dia(Dia matrix) nogil:
  pass

cpdef double mean_abs_dense(Dense matrix) nogil:
  pass
