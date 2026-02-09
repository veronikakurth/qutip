import pytest
import numpy as np
import qutip as qt
import numbers

from qutip.core.data.mean import mean_csr, mean_dia, mean_dense
from qutip.core.data.mean import mean_abs_csr, mean_abs_dia, mean_abs_dense
from qutip.core.data import CSR, Dia, Dense
from . import test_mathematics as testing


class TestMean(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        nnz = np.count_nonzero(matrix)

        if nnz == 0:
            return 0.0

        mean_val = matrix.sum() / nnz
        return mean_val

    specialisations = [
        pytest.param(mean_csr, CSR, numbers.Complex),
        pytest.param(mean_dia, Dia, numbers.Complex),
        pytest.param(mean_dense, Dense, numbers.Complex),
    ]

class TestAbsMean(testing.UnaryOpMixin):
    def op_numpy(self, matrix):
        nnz = np.count_nonzero(matrix)

        if nnz == 0:
            return 0.0

        abs_matrix = np.abs(matrix)
        mean_val = abs_matrix.sum() / nnz
        return mean_val

    specialisations = [
        pytest.param(mean_abs_csr, CSR, numbers.Complex),
        pytest.param(mean_abs_dia, Dia, numbers.Complex),
        pytest.param(mean_abs_dense, Dense, numbers.Complex),
    ]
