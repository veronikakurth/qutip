import pytest
import numpy as np
import qutip as qt
from qutip.core.data.mean import mean_csr, mean_dia
from qutip.core.data import CSR, Dia


# Fixtures to be reused
# TODO: this can belong the parametrization block and sub/super diagonal offsets can be computed based on dim
@pytest.fixture
def dim():
  return 5

@pytest.fixture
def main_diag_offset():
  return 0

@pytest.fixture
def sub_diag_offset():
  return -1

@pytest.fixture
def super_diag_offset():
  return 1

@pytest.fixture
def diag_elements():
  return [1, 2, 3, 4, 5]

@pytest.fixture
def sub_diag_elements():
  return [1, 2, 3, 4]

@pytest.fixture
def super_diag_elements():
  return [1, 2, 3, 4]

@pytest.fixture
def true_mean(diag_elements, sub_diag_elements, super_diag_elements):
  matrix = [diag_elements, sub_diag_elements, super_diag_elements]
  all_elements = [val for diag in matrix for val in diag]
  return np.mean(all_elements)

# ------ Dia matrices 

def main_diagonal():
  pass

def main_and_super_diagonal():
  pass

def main_and_sub_diagonal():
  pass

def sub_diagonal():
  pass

def super_diagonal():
  pass

def super_sub_diagonal():
  pass

# ------ CSR matrices 

# TODO: treat special cases like operating on differences/fractions/sums of matrices to check for numerical noise

def test_csr_matrix():
  N = 3

  data = np.array([4.0, -4.0, 1.0], dtype=complex)
  col_index = np.array([0, 2, 2], dtype=np.int32)
  row_index = np.array([0, 1, 2, 3], dtype=np.int32)

  matrix = CSR((data, col_index, row_index), shape=(N, N))

  expected = 1.0/3.0 #np.mean(data)
  result = mean_csr(matrix)
  assert np.isclose(result, expected, qt.settings.core['atol'])


# Dia matrices

def test_dia_matrix_main_diag():
  N = 3

  diag_data = np.array([1, 1, 2], dtype=complex)
  offsets = np.array([0], dtype=np.int32)

  matrix = Dia((diag_data, offsets), shape=(N, N))

  expected = 4/3
  result = mean_dia(matrix)

  assert np.isclose(result, expected, qt.settings.core['atol'])


# ------ Dense matrices 

def dense_matrix():
  pass



